// Package harmony renders and incrementally parses the Harmony chat format.
// It handles Harmony's control tokens only; text tokenization is intentionally
// left to the model's tokenizer.
package harmony

import "strings"

const (
	StartTokenID     uint32 = 199998
	EndTokenID       uint32 = 200000
	ReturnTokenID    uint32 = 200002
	ConstrainTokenID uint32 = 200003
	ChannelTokenID   uint32 = 200005
	MessageTokenID   uint32 = 200008
	CallTokenID      uint32 = 200012
)

const (
	StartToken     = "<|start|>"
	EndToken       = "<|end|>"
	ReturnToken    = "<|return|>"
	ConstrainToken = "<|constrain|>"
	ChannelToken   = "<|channel|>"
	MessageToken   = "<|message|>"
	CallToken      = "<|call|>"
)

// SpecialTokens maps Harmony control token text to its fixed token ID.
var SpecialTokens = map[string]uint32{
	StartToken:     StartTokenID,
	EndToken:       EndTokenID,
	ReturnToken:    ReturnTokenID,
	ConstrainToken: ConstrainTokenID,
	ChannelToken:   ChannelTokenID,
	MessageToken:   MessageTokenID,
	CallToken:      CallTokenID,
}

// Message is one system, developer, user, or assistant Harmony message.
// Channel is normally set only on assistant messages.
type Message struct {
	Role    string
	Channel Channel
	Content string
}

// RenderPrompt renders messages followed by an optional final user message and
// opens the assistant turn for generation. System and developer messages are
// supplied in messages, preserving their order and content verbatim.
func RenderPrompt(messages []Message, user string) string {
	var b strings.Builder
	for _, message := range messages {
		renderMessage(&b, message)
	}
	if user != "" {
		renderMessage(&b, Message{Role: "user", Content: user})
	}
	b.WriteString(StartToken)
	b.WriteString("assistant")
	return b.String()
}

func renderMessage(b *strings.Builder, message Message) {
	b.WriteString(StartToken)
	b.WriteString(message.Role)
	if message.Channel != "" {
		b.WriteString(ChannelToken)
		b.WriteString(string(message.Channel))
	}
	b.WriteString(MessageToken)
	b.WriteString(message.Content)
	b.WriteString(EndToken)
}
