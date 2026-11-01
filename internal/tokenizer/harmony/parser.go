package harmony

import (
	"errors"
	"strings"
)

// Channel is a Harmony assistant output channel.
type Channel string

const (
	ChannelAnalysis   Channel = "analysis"
	ChannelCommentary Channel = "commentary"
	ChannelFinal      Channel = "final"
)

// EventType identifies an incremental parser event.
type EventType uint8

const (
	EventText EventType = iota
	EventChannelStart
	EventChannelEnd
	EventStop
)

// Event is emitted as soon as its text or control token is complete. Stop
// events carry ReturnTokenID or CallTokenID in TokenID.
type Event struct {
	Type    EventType
	Channel Channel
	Text    string
	TokenID uint32
}

// Options controls which assistant channels produce visible text events.
type Options struct {
	// IncludeAnalysis exposes analysis text. Analysis is hidden by default.
	IncludeAnalysis bool
}

// ErrParserDone is returned when input is written after return or call.
var ErrParserDone = errors.New("harmony: parser is done")

// Parser incrementally decodes Harmony output. It is not safe for concurrent
// use. A Parser accepts either decoded text or token IDs with decoded text.
type Parser struct {
	options Options
	mode    parserMode
	channel Channel
	pending string
	done    bool
}

type parserMode uint8

const (
	modeHeader parserMode = iota
	modeChannel
	modeContent
)

// NewParser constructs an incremental Harmony parser.
func NewParser(options Options) *Parser {
	return &Parser{options: options}
}

// Done reports whether a return or call stop token has been consumed.
func (p *Parser) Done() bool {
	return p.done
}

// Write consumes one model token. For Harmony special tokens, text may be
// empty; for ordinary tokens, tokenID is ignored and text is parsed. Known
// special-token text is not parsed a second time.
func (p *Parser) Write(tokenID uint32, text string) ([]Event, error) {
	if p.done {
		return nil, ErrParserDone
	}
	if marker, ok := markerForID(tokenID); ok {
		return p.consumeMarker(marker), nil
	}
	return p.WriteText(text)
}

// WriteText consumes decoded text. Control tokens may be split at any byte
// boundary, including inside UTF-8 text, without exposing partial markers.
func (p *Parser) WriteText(text string) ([]Event, error) {
	if p.done {
		return nil, ErrParserDone
	}
	p.pending += text
	var events []Event
	for !p.done {
		marker, at := nextMarker(p.pending)
		if marker == "" {
			if p.mode == modeContent {
				keep := partialMarkerSuffix(p.pending)
				cut := len(p.pending) - keep
				events = append(events, p.consumeText(p.pending[:cut])...)
				p.pending = p.pending[cut:]
			}
			break
		}
		header := p.pending[:at]
		events = append(events, p.consumeText(header)...)
		p.pending = p.pending[at+len(marker):]
		events = append(events, p.applyMarker(marker, header)...)
	}
	return events, nil
}

func (p *Parser) consumeMarker(marker string) []Event {
	header := p.pending
	events := p.consumeText(header)
	p.pending = ""
	return append(events, p.applyMarker(marker, header)...)
}

func (p *Parser) consumeText(text string) []Event {
	if text == "" {
		return nil
	}
	if p.mode != modeContent || !p.visible(p.channel) {
		return nil
	}
	return []Event{{Type: EventText, Channel: p.channel, Text: text}}
}

func (p *Parser) applyMarker(marker, header string) []Event {
	switch marker {
	case StartToken:
		p.mode = modeHeader
		p.channel = ""
	case ChannelToken:
		p.mode = modeChannel
	case MessageToken:
		if p.mode == modeChannel {
			p.channel = Channel(strings.TrimSpace(header))
		}
		p.mode = modeContent
		if p.channel != "" {
			return []Event{{Type: EventChannelStart, Channel: p.channel}}
		}
	case EndToken:
		p.mode = modeHeader
		return p.endChannel()
	case ReturnToken, CallToken:
		channel := p.channel
		events := p.endChannel()
		p.mode = modeHeader
		p.done = true
		id := ReturnTokenID
		if marker == CallToken {
			id = CallTokenID
		}
		return append(events, Event{Type: EventStop, Channel: channel, TokenID: id})
	}
	return nil
}

func (p *Parser) endChannel() []Event {
	if p.channel == "" {
		return nil
	}
	channel := p.channel
	p.channel = ""
	return []Event{{Type: EventChannelEnd, Channel: channel}}
}

func (p *Parser) visible(channel Channel) bool {
	return channel != ChannelAnalysis || p.options.IncludeAnalysis
}

func markerForID(id uint32) (string, bool) {
	for marker, markerID := range SpecialTokens {
		if id == markerID {
			return marker, true
		}
	}
	return "", false
}

func nextMarker(text string) (string, int) {
	bestAt := -1
	var best string
	for marker := range SpecialTokens {
		at := strings.Index(text, marker)
		if at >= 0 && (bestAt < 0 || at < bestAt) {
			best, bestAt = marker, at
		}
	}
	return best, bestAt
}

func partialMarkerSuffix(text string) int {
	max := 0
	for marker := range SpecialTokens {
		limit := len(marker) - 1
		if len(text) < limit {
			limit = len(text)
		}
		for size := limit; size > max; size-- {
			if strings.HasSuffix(text, marker[:size]) {
				max = size
				break
			}
		}
	}
	return max
}
