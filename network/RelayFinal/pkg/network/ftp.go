package network

import (
	"bytes"
	"fmt"
	"image"
	"image/jpeg"
	"image/png"
	"os"
)

func ImageToBytes(filePath string, format string) ([]byte, error) {
	// Open image file
	file, err := os.Open(filePath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	// Decode image
	img, _, err := image.Decode(file)
	if err != nil {
		return nil, err
	}

	// Encode into buffer
	var buf bytes.Buffer
	switch format {
	case "png":
		err = png.Encode(&buf, img)
	case "jpg", "jpeg":
		err = jpeg.Encode(&buf, img, nil) // you can pass options here
	default:
		return nil, fmt.Errorf("unsupported format: %s", format)
	}
	if err != nil {
		return nil, err
	}

	return buf.Bytes(), nil
}

func BytesToImage(data []byte, format string, outPath string) error {
	// Create reader from bytes
	r := bytes.NewReader(data)

	// Decode image from bytes
	img, _, err := image.Decode(r)
	if err != nil {
		return err
	}

	// Save back to file
	outFile, err := os.Create(outPath)
	if err != nil {
		return err
	}
	defer outFile.Close()

	switch format {
	case "png":
		err = png.Encode(outFile, img)
	case "jpg", "jpeg":
		err = jpeg.Encode(outFile, img, nil)
	default:
		return fmt.Errorf("unsupported format: %s", format)
	}

	return err
}

// func main() {
// 	// Convert to bytes
// 	data, err := imageToBytes("input.png", "png")
// 	if err != nil {
// 		panic(err)
// 	}
// 	fmt.Println("Image converted to byte array, length:", len(data))

// 	// Optionally, save the byte slice to a file
// 	if err := ioutil.WriteFile("image_bytes.bin", data, 0644); err != nil {
// 		panic(err)
// 	}

// 	// Convert bytes back to image
// 	if err := bytesToImage(data, "png", "output.png"); err != nil {
// 		panic(err)
// 	}
// 	fmt.Println("Bytes written back as output.png")
// }
