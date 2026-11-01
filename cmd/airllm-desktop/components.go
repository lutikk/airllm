package main

import (
	"fyne.io/fyne/v2"
	"fyne.io/fyne/v2/canvas"
	"fyne.io/fyne/v2/container"
	"fyne.io/fyne/v2/layout"
	"fyne.io/fyne/v2/theme"
	"fyne.io/fyne/v2/widget"
)

func styledCard(title, subtitle string, content fyne.CanvasObject) fyne.CanvasObject {
	background := canvas.NewRectangle(theme.Color(theme.ColorNameInputBackground))
	background.CornerRadius = 8
	background.StrokeColor = theme.Color(theme.ColorNameSeparator)
	background.StrokeWidth = 1

	titleText := canvas.NewText(title, theme.Color(theme.ColorNameForeground))
	titleText.TextSize = 15
	titleText.TextStyle = fyne.TextStyle{Bold: true}

	var header fyne.CanvasObject = titleText
	if subtitle != "" {
		subtitleLabel := widget.NewLabel(subtitle)
		subtitleLabel.Wrapping = fyne.TextWrapWord
		subtitleLabel.Importance = widget.LowImportance
		header = container.NewVBox(titleText, subtitleLabel)
	}

	inner := container.New(
		layout.NewCustomPaddedLayout(12, 12, 16, 16),
		container.NewVBox(header, content),
	)
	return container.NewStack(background, inner)
}

func sectionLabel(text string) *widget.Label {
	label := widget.NewLabel(text)
	label.Importance = widget.LowImportance
	label.Wrapping = fyne.TextWrapWord
	return label
}

func opticallyCentered(object fyne.CanvasObject) fyne.CanvasObject {
	return container.New(layout.NewCustomPaddedLayout(3, 0, 0, 0), object)
}

type fixedHeightLayout struct {
	height float32
}

func (l fixedHeightLayout) MinSize(objects []fyne.CanvasObject) fyne.Size {
	width := float32(0)
	for _, object := range objects {
		width = max(width, object.MinSize().Width)
	}
	return fyne.NewSize(width, l.height)
}

func (l fixedHeightLayout) Layout(objects []fyne.CanvasObject, size fyne.Size) {
	for _, object := range objects {
		object.Resize(fyne.NewSize(size.Width, size.Height))
		object.Move(fyne.NewPos(0, 0))
	}
}
