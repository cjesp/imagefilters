package imagefilters

import "math"

func gaussMask(stdDev float64, x, y int) float64 {
	stdDevSquarred := stdDev*stdDev
	temp := -1.0 * (float64(x*x + y*y) / (2.0*stdDevSquarred))
	e := math.Exp(temp)
	leftSide := 2.0*math.Pi*stdDevSquarred
	return e / leftSide
}

func gaussKernel(stdDev float64, size int) [][]float64 {
	kernelSize := 2 * size + 1
	gauss := make([][]float64, kernelSize)
	for i := range gauss {
		gauss[i] = make([]float64, kernelSize)
	}

	//calculating gauss mask
	gaussSum := 0.0
	for x := -size; x < size + 1; x++ {
		for y := -size; y < size + 1 ; y++ {
			gaussVal := gaussMask(stdDev, x, y)
			gauss[x+size][y+size] = gaussVal
			gaussSum += gaussVal
		}
	}

	// normalizing gauss mask to sum 1
	for x := -size; x < size + 1; x++ {
		for y := -size; y < size + 1 ; y++ {
			gauss[x+size][y+size] /= gaussSum
		}
	}

	return gauss
}

func gaussianBlurKernel1D(x int, sigma float64) float64 {
	return math.Exp(-float64(x*x)/(2*sigma*sigma)) / (sigma * math.Sqrt(2*math.Pi))
}