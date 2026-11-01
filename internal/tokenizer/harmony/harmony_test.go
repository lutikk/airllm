package harmony_test

import (
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/lutikk/airllm/internal/tokenizer/harmony"
)

func TestRenderPrompt(t *testing.T) {
	messages := []harmony.Message{
		{Role: "system", Content: "You are helpful."},
		{Role: "developer", Content: "# Instructions\n\nBe concise."},
		{Role: "user", Content: "Earlier question"},
		{Role: "assistant", Channel: harmony.ChannelFinal, Content: "Earlier answer"},
	}
	got := harmony.RenderPrompt(messages, "Current question")
	want := "<|start|>system<|message|>You are helpful.<|end|>" +
		"<|start|>developer<|message|># Instructions\n\nBe concise.<|end|>" +
		"<|start|>user<|message|>Earlier question<|end|>" +
		"<|start|>assistant<|channel|>final<|message|>Earlier answer<|end|>" +
		"<|start|>user<|message|>Current question<|end|>" +
		"<|start|>assistant"
	if got != want {
		t.Fatalf("RenderPrompt() =\n%q\nwant:\n%q", got, want)
	}
}

func TestRenderPromptWithoutAdditionalUserMessage(t *testing.T) {
	got := harmony.RenderPrompt(nil, "")
	if want := "<|start|>assistant"; got != want {
		t.Fatalf("RenderPrompt() = %q, want %q", got, want)
	}
}

func TestParserSplitTokensAndChannels(t *testing.T) {
	stream := "<|start|>assistant<|channel|>analysis<|message|>secret reasoning" +
		"<|end|><|start|>assistant<|channel|>commentary<|message|>tool input" +
		"<|end|><|start|>assistant<|channel|>final<|message|>visible answer<|return|>"
	parser := harmony.NewParser(harmony.Options{})
	var events []harmony.Event
	for _, fragment := range splitEveryByte(stream) {
		got, err := parser.WriteText(fragment)
		if err != nil {
			t.Fatalf("WriteText(%q) error = %v", fragment, err)
		}
		events = append(events, got...)
	}

	if got, want := visibleText(events, harmony.ChannelFinal), "visible answer"; got != want {
		t.Fatalf("final text = %q, want %q", got, want)
	}
	if got, want := visibleText(events, harmony.ChannelCommentary), "tool input"; got != want {
		t.Fatalf("commentary text = %q, want %q", got, want)
	}
	if got := visibleText(events, harmony.ChannelAnalysis); got != "" {
		t.Fatalf("analysis text = %q, want hidden", got)
	}
	assertChannelLifecycle(t, events, harmony.ChannelAnalysis)
	assertChannelLifecycle(t, events, harmony.ChannelCommentary)
	assertChannelLifecycle(t, events, harmony.ChannelFinal)
	last := events[len(events)-1]
	if last.Type != harmony.EventStop || last.TokenID != harmony.ReturnTokenID || last.Channel != harmony.ChannelFinal {
		t.Fatalf("last event = %+v, want final return stop", last)
	}
	if !parser.Done() {
		t.Fatal("Done() = false after return")
	}
}

func TestParserCanExposeAnalysis(t *testing.T) {
	parser := harmony.NewParser(harmony.Options{IncludeAnalysis: true})
	events, err := parser.WriteText("<|start|>assistant<|channel|>analysis<|message|>reasoning<|return|>")
	if err != nil {
		t.Fatal(err)
	}
	if got := visibleText(events, harmony.ChannelAnalysis); got != "reasoning" {
		t.Fatalf("analysis text = %q, want reasoning", got)
	}
}

func TestParserTokenIDsAndCallStop(t *testing.T) {
	parser := harmony.NewParser(harmony.Options{})
	writes := []struct {
		id   uint32
		text string
	}{
		{harmony.StartTokenID, ""},
		{0, "assistant"},
		{harmony.ChannelTokenID, ""},
		{0, "commentary"},
		{harmony.MessageTokenID, ""},
		{0, "calling tool"},
		{harmony.CallTokenID, ""},
	}
	var events []harmony.Event
	for _, write := range writes {
		got, err := parser.Write(write.id, write.text)
		if err != nil {
			t.Fatalf("Write(%d, %q) error = %v", write.id, write.text, err)
		}
		events = append(events, got...)
	}
	if got := visibleText(events, harmony.ChannelCommentary); got != "calling tool" {
		t.Fatalf("commentary text = %q, want calling tool", got)
	}
	last := events[len(events)-1]
	if last.Type != harmony.EventStop || last.TokenID != harmony.CallTokenID {
		t.Fatalf("last event = %+v, want call stop", last)
	}
	if _, err := parser.Write(0, "ignored"); !errors.Is(err, harmony.ErrParserDone) {
		t.Fatalf("Write() after stop error = %v, want ErrParserDone", err)
	}
}

func TestParserDoesNotLeakPartialSpecialToken(t *testing.T) {
	parser := harmony.NewParser(harmony.Options{})
	events, err := parser.WriteText("<|start|>assistant<|channel|>final<|message|>safe<|ret")
	if err != nil {
		t.Fatal(err)
	}
	if got := strings.Join(textEvents(events), ""); got != "safe" {
		t.Fatalf("text before completed marker = %q, want safe", got)
	}
	events, err = parser.WriteText("urn|>")
	if err != nil {
		t.Fatal(err)
	}
	if got := textEvents(events); len(got) != 0 {
		t.Fatalf("stop marker leaked as text: %q", got)
	}
	if last := events[len(events)-1]; last.Type != harmony.EventStop || last.TokenID != harmony.ReturnTokenID {
		t.Fatalf("last event = %+v, want return stop", last)
	}
}

func TestSpecialTokenIDs(t *testing.T) {
	want := map[string]uint32{
		"<|start|>": 199998, "<|end|>": 200000, "<|return|>": 200002,
		"<|constrain|>": 200003, "<|channel|>": 200005,
		"<|message|>": 200008, "<|call|>": 200012,
	}
	if !reflect.DeepEqual(harmony.SpecialTokens, want) {
		t.Fatalf("SpecialTokens = %v, want %v", harmony.SpecialTokens, want)
	}
}

func splitEveryByte(text string) []string {
	parts := make([]string, len(text))
	for i := range text {
		parts[i] = text[i : i+1]
	}
	return parts
}

func visibleText(events []harmony.Event, channel harmony.Channel) string {
	var b strings.Builder
	for _, event := range events {
		if event.Type == harmony.EventText && event.Channel == channel {
			b.WriteString(event.Text)
		}
	}
	return b.String()
}

func textEvents(events []harmony.Event) []string {
	var result []string
	for _, event := range events {
		if event.Type == harmony.EventText {
			result = append(result, event.Text)
		}
	}
	return result
}

func assertChannelLifecycle(t *testing.T, events []harmony.Event, channel harmony.Channel) {
	t.Helper()
	var starts, ends int
	for _, event := range events {
		if event.Channel != channel {
			continue
		}
		switch event.Type {
		case harmony.EventChannelStart:
			starts++
		case harmony.EventChannelEnd:
			ends++
		}
	}
	if starts != 1 || ends != 1 {
		t.Fatalf("channel %q lifecycle starts=%d ends=%d, want 1 each", channel, starts, ends)
	}
}
