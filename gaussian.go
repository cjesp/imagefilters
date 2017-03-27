package imagefilters

import (
	"image"
	"sync"
	"image/color"
	"math"
	"fmt"
)

var wg sync.WaitGroup

// GaussianBlur generates a blurred image based on it's parameters
func GaussianBlur(stdDev float64, maskSize, workers int, src image.Image) (blurredImg image.Image, err error)  {
	defer wg.Wait()
	
	gauss := gaussKernel(stdDev, maskSize)
	xLength, yLength := src.Bounds().Max.X, src.Bounds().Max.Y
	remainder := xLength % workers
	step := xLength / workers
	out := image.NewRGBA(src.Bounds())
	wg.Add(workers)

	for i := 0; i < workers; i++ {
		if i == workers - 1 {
			go parallelGaussianBlur(i * step, i * step + step + remainder, yLength, src, out, gauss)	
			continue
		}

		go parallelGaussianBlur(i * step, i * step + step, yLength, src, out, gauss)		
	}

	return out, nil
}

func parallelGaussianBlur(xStart, xEnd, yLength int,  src image.Image, out *image.RGBA, mask [][]float64)  {
	defer wg.Done()
	for x := xStart; x < xEnd; x++ {
		for y := 0; y < yLength; y++ {
			getGaussianBlurArray(x, y, out.Stride, src, out.Pix, mask)
		}
	}
}

func getGaussianBlur(x, y int, src image.Image, out *image.RGBA, mask [][]float64) {
	length := (len(mask) - 1) / 2
	x0, y0, x1, y1 := src.Bounds().Min.X, src.Bounds().Min.Y, src.Bounds().Max.X, src.Bounds().Max.Y
	R, G, B := 0.0, 0.0, 0.0
	xM, yM := 0, 0
	for xI := x - length; xI < x + length; xI++ {
		xM = xI - x + length
		for yI := y - length; yI < y + length; yI++ {
			if xI < x0 || xI > x1 || yI < y0 || yI > y1 {
				continue
			}
			yM = yI - y + length

			r, g, b, _ := src.At(xI, yI).RGBA()
			// if xM >= len(mask) || yM >= len(mask) {
			// 	fmt.Printf("xM=%v, yM=%v, xI=%v, yI=%v, x=%v, y=%v\n", xM, yM, xI, yI, x, y)
			// }

			R = R + float64(r / 256) * mask[xM][yM]
			G = G + float64(g / 256) * mask[xM][yM]
			B = B + float64(b / 256) * mask[xM][yM]
			
		}
	}

	pixel := color.RGBA{R: uint8(R), G: uint8(G), B: uint8(B), A: 255} 
	out.Set(x, y, pixel)

}

func getGaussianBlurArray(x, y, stride int, src image.Image, out []uint8, mask [][]float64)  {
	length := (len(mask) - 1) / 2
	x0, y0, x1, y1 := src.Bounds().Min.X, src.Bounds().Min.Y, src.Bounds().Max.X, src.Bounds().Max.Y
	R, G, B := 0.0, 0.0, 0.0
	xM, yM := 0, 0
	for xI := x - length; xI < x + length; xI++ {
		xM = xI - x + length
		for yI := y - length; yI < y + length; yI++ {
			if xI < x0 || xI > x1 || yI < y0 || yI > y1 {
				continue
			}
			yM = yI - y + length

			r, g, b, _ := src.At(xI, yI).RGBA()
			R = R + float64(r / 256) * mask[xM][yM]
			G = G + float64(g / 256) * mask[xM][yM]
			B = B + float64(b / 256) * mask[xM][yM]
			
		}
	}
	
	index := (x + y * y1) * 4
	out[index] = uint8(R)
	out[index+1] = uint8(G)
	out[index+2] = uint8(B)
	out[index+3] = uint8(255) 
}

//////////////////////////////////////////////////////////////

// GaussianBlur1D generates a one dimensional gauss kernel
func GaussianBlur1D(sigma float64, src image.Image) *image.RGBA  {
	kernelSize := int(math.Ceil(sigma * 3.0))
	// out := image.NewRGBA(src.Bounds())

	if kernelSize % 2 != 0 {
		kernelSize++
	}

	radius := kernelSize / 2
	kernel := make([]float64, kernelSize + 1)

	for i := 0; i < len(kernel); i++ {
		kernel[i] = gaussianBlurKernel1D(i - radius, sigma)
	}

	var gaussSum float64
	for _, value := range kernel {
		gaussSum += value
	}
	
	for index := range kernel {
		kernel[index] /= gaussSum
	}



	img := horizontalBlur(src, kernel)
	finalOuput := verticalBlur(img, kernel)
	

	return finalOuput
	
}

func horizontalBlur(src image.Image, kernel []float64) *image.RGBA  {
	bounds := src.Bounds()
	radius := (len(kernel) - 1) / 2
	img := image.NewRGBA(src.Bounds())

	for x := 0; x < bounds.Max.X; x++ {
		for y := 0; y < bounds.Max.Y; y++ {
			var R,G,B,A float64
			index := (x + y * bounds.Max.Y) * 4
			
			for i := -radius; i < radius + 1; i++ {
				if i + x < 0 || i + x > bounds.Max.X - 1 {
					continue
				}
				r,g,b,a := src.At(i+x, y).RGBA()
				kernelVal := kernel[i+radius]
				R += float64(r) * kernelVal 
				G += float64(g) * kernelVal
				B += float64(b) * kernelVal
				A += float64(a) * kernelVal
			}

			test := false
			if index >= 512*512*4 {
				test = true
			}

			if test {
				fmt.Println("overflow")
			}

			img.Pix[index] = uint8(R / 256.0)
			img.Pix[index+1] = uint8(G / 256.0)
			img.Pix[index+2] = uint8(B / 256.0)
			img.Pix[index+3] = 255
		}
	}
	return img

}

func verticalBlur(src image.Image, kernel []float64) *image.RGBA {
	bounds := src.Bounds()
	radius := (len(kernel) - 1) / 2
	img := image.NewRGBA(src.Bounds())

	for x := 0; x < bounds.Max.X; x++ {
		for y := 0; y < bounds.Max.Y; y++ {
			var R,G,B,A float64
			index := (x + y * bounds.Max.Y) * 4
			// r,g,b,a := src.At(x, y).RGBA()

			
			for i := -radius; i < radius + 1; i++ {
				if i + y < 0 || i + y > bounds.Max.Y - 1 {
					continue
				}
				r,g,b,a := src.At(x, i+y).RGBA()
				kernelVal := kernel[i+radius]
				R += float64(r) * kernelVal 
				G += float64(g) * kernelVal
				B += float64(b) * kernelVal
				A += float64(a) * kernelVal
			}

			img.Pix[index] += uint8(R / 256.0)
			img.Pix[index+1] += uint8(G / 256.0)
			img.Pix[index+2] += uint8(B / 256.0)
			img.Pix[index+3] = 255
		}
	}

	return img
}

func parallel(height, fn func(partStart, partEnd int))  {
	
}


