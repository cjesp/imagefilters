package imagefilters

import (
	"image"
	"sync"
	"image/color"
	"math"
	"runtime"
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
func GaussianBlur1D(sigma float64, src image.Image) *image.NRGBA  {
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

	src = toNRGBA(src)

	img := horizontalBlur(src, kernel)
	finalOuput := verticalBlur(img, kernel)
	

	return finalOuput
	
}

func horizontalBlur(src image.Image, kernel []float64) *image.NRGBA  {
	bounds := src.Bounds()
	radius := (len(kernel) - 1) / 2
	img := image.NewNRGBA(bounds)

	parallel(bounds.Max.X, func(start, end int){
		for x := start; x < end; x++ {
			for y := 0; y < bounds.Max.Y; y++ {
				// fmt.Printf("xStart=%v, xEnd=%v\n", start, end)
				var R,G,B,A float64
				index := (x + y * bounds.Max.X) * 4

				// if index == 28212640 || (x == 3820 && y == 1840) {
				// 	fmt.Println("28212640 was hit")
				// }
				
				for i := -radius; i < radius + 1; i++ {
					if i + x < 0 || i + x > bounds.Max.X - 1 {
						continue
					}
					r,g,b,a := src.At(i+x, y).RGBA() 
					kernelVal := kernel[i+radius]
					R += float64(r / 256) * kernelVal 
					G += float64(g / 256) * kernelVal
					B += float64(b / 256) * kernelVal
					A += float64(a / 256) * kernelVal
				}

				// test := false
				// if index >= 512*512*4 {
				// 	test = true
				// }

				// if test {
				// 	fmt.Println("overflow")
				// }

				img.Pix[index] = uint8(R)
				img.Pix[index+1] = uint8(G)
				img.Pix[index+2] = uint8(B)
				img.Pix[index+3] = 255
			}
		}
	})

	return img

}

func verticalBlur(src image.Image, kernel []float64) *image.NRGBA {
	bounds := src.Bounds()
	radius := (len(kernel) - 1) / 2
	img := image.NewNRGBA(bounds)

	parallel(bounds.Max.X, func(start, end int) {
		for x := start; x < end; x++ {
			for y := 0; y < bounds.Max.Y; y++ {
				var R,G,B,A float64
				index := (x + y * bounds.Max.X) * 4
				// r,g,b,a := src.At(x, y).RGBA()

				
				for i := -radius; i < radius + 1; i++ {
					if i + y < 0 || i + y > bounds.Max.Y - 1 {
						continue
					}
					r,g,b,a := src.At(x, i+y).RGBA()
					kernelVal := kernel[i+radius]
					R += float64(r / 256) * kernelVal 
					G += float64(g / 256) * kernelVal
					B += float64(b / 256) * kernelVal
					A += float64(a / 256) * kernelVal
				}

				img.Pix[index] += uint8(R)
				img.Pix[index+1] += uint8(G)
				img.Pix[index+2] += uint8(B)
				img.Pix[index+3] = 255
			}
		}
	})

	return img
}

func parallel(height int, fn func(partStart, partEnd int))  {
	cores := runtime.GOMAXPROCS(0)
	goRoutines := height / cores

	if cores == 1 {
		fn(0, height)
	} else {
		var wg sync.WaitGroup
		wg.Add(cores)
		defer wg.Wait()

		for i := 0; i < cores; i++ {
			// last step
			if i == cores - 1 {
				go func(start, end int) {
					defer wg.Done()
				 	fn(start, end)
				}(goRoutines * i, height)
			} else {
				go func(start, end int) {
					defer wg.Done()
					fn(start, end)
				}(goRoutines * i, goRoutines * (i + 1))
			}
		}
	}
}

// func BilateralFilter(sigmaS, sigmaI float64, src image.Image) *image.RGBA  {
// 	radiusS, radiusR := math.Ceil()
// 	kernselS, kernelI := generate1DGaussKernel()	
// }


