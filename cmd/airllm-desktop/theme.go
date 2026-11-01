package main

import (
	"image/color"

	"fyne.io/fyne/v2"
	"fyne.io/fyne/v2/theme"
)

// darkTheme реализует тёмную тему Slate Indigo из дизайн-системы AutoPixel.
type darkTheme struct{}

var _ fyne.Theme = (*darkTheme)(nil)

var (
	colBackground       = color.NRGBA{R: 0x0E, G: 0x11, B: 0x16, A: 0xFF}
	colSurface          = color.NRGBA{R: 0x1A, G: 0x1E, B: 0x27, A: 0xFF}
	colHeaderBackground = color.NRGBA{R: 0x16, G: 0x19, B: 0x22, A: 0xFF}
	colOverlay          = color.NRGBA{R: 0x0E, G: 0x11, B: 0x16, A: 0xF0}
	colSeparator        = color.NRGBA{R: 0x23, G: 0x27, B: 0x35, A: 0xFF}

	colForeground  = color.NRGBA{R: 0xE8, G: 0xEA, B: 0xED, A: 0xFF}
	colDisabled    = color.NRGBA{R: 0x6B, G: 0x72, B: 0x80, A: 0xFF}
	colPlaceholder = color.NRGBA{R: 0x5C, G: 0x62, B: 0x73, A: 0xFF}
	colShadow      = color.NRGBA{R: 0x00, G: 0x00, B: 0x00, A: 0x99}

	colPrimary    = color.NRGBA{R: 0x7C, G: 0x5C, B: 0xFF, A: 0xFF}
	colPrimaryAlt = color.NRGBA{R: 0x68, G: 0x45, B: 0xE6, A: 0xFF}
	colHover      = color.NRGBA{R: 0x7C, G: 0x5C, B: 0xFF, A: 0x26}
	colFocus      = color.NRGBA{R: 0x7C, G: 0x5C, B: 0xFF, A: 0x73}
	colSelection  = color.NRGBA{R: 0x7C, G: 0x5C, B: 0xFF, A: 0x40}
	colScrollBar  = color.NRGBA{R: 0x7C, G: 0x5C, B: 0xFF, A: 0x80}

	colSuccess = color.NRGBA{R: 0x34, G: 0xD3, B: 0x99, A: 0xFF}
	colWarning = color.NRGBA{R: 0xFB, G: 0xBF, B: 0x24, A: 0xFF}
	colError   = color.NRGBA{R: 0xF8, G: 0x71, B: 0x71, A: 0xFF}
)

func (t *darkTheme) Color(name fyne.ThemeColorName, variant fyne.ThemeVariant) color.Color {
	switch name {
	case theme.ColorNameBackground:
		return colBackground
	case theme.ColorNameButton, theme.ColorNamePrimary:
		return colPrimary
	case theme.ColorNameFocus:
		return colFocus
	case theme.ColorNameForeground:
		return colForeground
	case theme.ColorNameDisabled, theme.ColorNameDisabledButton:
		return colDisabled
	case theme.ColorNameInputBackground, theme.ColorNameMenuBackground:
		return colSurface
	case theme.ColorNameInputBorder, theme.ColorNameSeparator:
		return colSeparator
	case theme.ColorNameShadow:
		return colShadow
	case theme.ColorNameHeaderBackground:
		return colHeaderBackground
	case theme.ColorNameScrollBar:
		return colScrollBar
	case theme.ColorNameHover:
		return colHover
	case theme.ColorNamePlaceHolder:
		return colPlaceholder
	case theme.ColorNamePressed:
		return colPrimaryAlt
	case theme.ColorNameSelection:
		return colSelection
	case theme.ColorNameSuccess:
		return colSuccess
	case theme.ColorNameWarning:
		return colWarning
	case theme.ColorNameError:
		return colError
	case theme.ColorNameOverlayBackground:
		return colOverlay
	}
	return theme.DefaultTheme().Color(name, variant)
}

func (t *darkTheme) Font(style fyne.TextStyle) fyne.Resource {
	return theme.DefaultTheme().Font(style)
}

func (t *darkTheme) Icon(name fyne.ThemeIconName) fyne.Resource {
	return theme.DefaultTheme().Icon(name)
}

func (t *darkTheme) Size(name fyne.ThemeSizeName) float32 {
	switch name {
	case theme.SizeNamePadding:
		return 8
	case theme.SizeNameInnerPadding:
		return 12
	case theme.SizeNameText:
		return 13
	case theme.SizeNameHeadingText:
		return 20
	case theme.SizeNameSubHeadingText:
		return 15
	case theme.SizeNameCaptionText:
		return 11
	case theme.SizeNameInputRadius:
		return 10
	case theme.SizeNameSelectionRadius:
		return 8
	case theme.SizeNameScrollBar:
		return 6
	case theme.SizeNameScrollBarSmall:
		return 3
	case theme.SizeNameSeparatorThickness:
		return 1
	}
	return theme.DefaultTheme().Size(name)
}
