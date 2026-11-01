package main

import (
	"image/color"
	"testing"

	"fyne.io/fyne/v2"
	"fyne.io/fyne/v2/theme"
)

func TestSlateIndigoPalette(t *testing.T) {
	palette := &darkTheme{}
	tests := []struct {
		name fyne.ThemeColorName
		want color.NRGBA
	}{
		{theme.ColorNameBackground, color.NRGBA{R: 0x0E, G: 0x11, B: 0x16, A: 0xFF}},
		{theme.ColorNameInputBackground, color.NRGBA{R: 0x1A, G: 0x1E, B: 0x27, A: 0xFF}},
		{theme.ColorNameHeaderBackground, color.NRGBA{R: 0x16, G: 0x19, B: 0x22, A: 0xFF}},
		{theme.ColorNamePrimary, color.NRGBA{R: 0x7C, G: 0x5C, B: 0xFF, A: 0xFF}},
		{theme.ColorNameSeparator, color.NRGBA{R: 0x23, G: 0x27, B: 0x35, A: 0xFF}},
		{theme.ColorNameForeground, color.NRGBA{R: 0xE8, G: 0xEA, B: 0xED, A: 0xFF}},
		{theme.ColorNameDisabled, color.NRGBA{R: 0x6B, G: 0x72, B: 0x80, A: 0xFF}},
		{theme.ColorNamePlaceHolder, color.NRGBA{R: 0x5C, G: 0x62, B: 0x73, A: 0xFF}},
		{theme.ColorNamePressed, color.NRGBA{R: 0x68, G: 0x45, B: 0xE6, A: 0xFF}},
		{theme.ColorNameSuccess, color.NRGBA{R: 0x34, G: 0xD3, B: 0x99, A: 0xFF}},
		{theme.ColorNameWarning, color.NRGBA{R: 0xFB, G: 0xBF, B: 0x24, A: 0xFF}},
		{theme.ColorNameError, color.NRGBA{R: 0xF8, G: 0x71, B: 0x71, A: 0xFF}},
	}
	for _, test := range tests {
		t.Run(string(test.name), func(t *testing.T) {
			got := color.NRGBAModel.Convert(palette.Color(test.name, theme.VariantDark)).(color.NRGBA)
			if got != test.want {
				t.Fatalf("Color(%s) = %#v; want %#v", test.name, got, test.want)
			}
		})
	}
}

func TestSlateIndigoTypographyAndSpacing(t *testing.T) {
	palette := &darkTheme{}
	tests := []struct {
		name fyne.ThemeSizeName
		want float32
	}{
		{theme.SizeNamePadding, 8},
		{theme.SizeNameInnerPadding, 12},
		{theme.SizeNameText, 13},
		{theme.SizeNameHeadingText, 20},
		{theme.SizeNameSubHeadingText, 15},
		{theme.SizeNameCaptionText, 11},
		{theme.SizeNameInputRadius, 10},
		{theme.SizeNameSelectionRadius, 8},
		{theme.SizeNameScrollBar, 6},
	}
	for _, test := range tests {
		if got := palette.Size(test.name); got != test.want {
			t.Errorf("Size(%s) = %v; want %v", test.name, got, test.want)
		}
	}
}
