package parseptron

import java.awt.image.BufferedImage
import kotlin.math.pow
import kotlin.random.Random

class Parceptron2(
    inputSize: Int = INPUTS_SIZE,
    firstLayerSize: Int = FIRST_LAYER_SIZE,
    secondLayerSize: Int = SECOND_LAYER_SIZE,
    outputSize: Int = OUTPUT_SIZE
) {

    companion object {

        //TODO Надо сделать так чтобы размеры сами заполнялись
        const val INPUTS_SIZE = 784 // кол-во пикселей на картинке
        const val FIRST_LAYER_SIZE = 16
        const val SECOND_LAYER_SIZE = 16
        const val OUTPUT_SIZE = 10

        const val COUNT_SUMMED_GRADIENTS = 100
        const val LEARNING_SPEED = 0.001f
    }

    val _gradientVectorSOWeights = mutableListOf<MutableList<Double>>()
    val _gradientVectorFSWeights = mutableListOf<MutableList<Double>>()
    val _gradientVectorIFWeights = mutableListOf<MutableList<Double>>()
    val _gradientVectorSOOffsets = mutableListOf<Double>()
    val _gradientVectorFSOffsets = mutableListOf<Double>()
    val _gradientVectorIFOffsets = mutableListOf<Double>()

    val inputLayerValues: Array<Double> = getRandomVector(inputSize)
    val inputWeightsValues: Array<Array<Double>> = getRandomMatrix(width = inputSize, height = firstLayerSize)
    val inputOffsetsValues: Array<Double> = getRandomVector(height = firstLayerSize)

    val firstLayer: Array<Double> = getRandomVector(firstLayerSize)
    val firstLayerWeightsValues: Array<Array<Double>> =
        getRandomMatrix(width = firstLayerSize, height = secondLayerSize)
    val firstLayerOffsetsValues: Array<Double> = getRandomVector(height = secondLayerSize)

    val secondLayer: Array<Double> = getRandomVector(secondLayerSize)
    val secondLayerWeightsValues: Array<Array<Double>> = getRandomMatrix(width = secondLayerSize, height = outputSize)
    val secondLayerOffsetsValues: Array<Double> = getRandomVector(height = outputSize)

    val outputLayer: Array<Double> = getRandomVector(outputSize)

    private fun getRandomVector(height: Int): Array<Double> {
        val vector = mutableListOf<Double>()

        for (j in 0 until height) {
            val value = getRandom()

            vector.add(value)
        }

        return vector.toTypedArray()
    }

    private fun getRandomMatrix(width: Int, height: Int): Array<Array<Double>> {
        val matrix = mutableListOf<Array<Double>>()

        for (i in 0 until height) {
            val row = mutableListOf<Double>()
            for (j in 0 until width) {
                val weight = getRandom()

                row.add(weight)
            }
            matrix.add(row.toTypedArray())
        }

        return matrix.toTypedArray()
    }

    private fun getRandom(): Double {
        val numberDigits = Random.nextInt(10) / 10.0
        val sign = if ((Random.nextInt(10) / 10.0) > 0.5) -1 else 1
        return Random.nextInt(10) * sign / 10.0
    }

    fun recognize(image: BufferedImage): Int {
        fillInputLayer(image)

        return activatingLayers()
    }

    private fun fillInputLayer(image: BufferedImage) {
        var inputCounter = 0

        if (image.height * image.width == inputLayerValues.count()) {
            for (y in 0 until image.height) {
                for (x in 0 until image.width) {
                    val color = image.getRGB(x, y)
                    inputLayerValues[inputCounter] = if (getPixelValue(color) > 0.3) 1.0 else 0.0
                    inputCounter++
                }
            }
        } else {
            throw IllegalArgumentException("Изображение должно быть 28*28 пикселей")
        }
    }

    private fun activatingLayers(): Int {
        activateNextLayer(
            currentLayer = inputLayerValues,
            currentLayerWeights = inputWeightsValues,
            currentLayerOffsets = inputOffsetsValues,
            nextLayer = firstLayer
        )

        activateNextLayer(
            currentLayer = firstLayer,
            currentLayerWeights = firstLayerWeightsValues,
            currentLayerOffsets = firstLayerOffsetsValues,
            nextLayer = secondLayer
        )

        activateNextLayer(
            currentLayer = secondLayer,
            currentLayerWeights = secondLayerWeightsValues,
            currentLayerOffsets = secondLayerOffsetsValues,
            nextLayer = outputLayer
        )

        return getResultNumber()
    }

    private fun getPixelValue(color: Int): Double {
        val red = (color ushr 16) and 0xFF
        val green = (color ushr 8) and 0xFF
        val blue = (color ushr 0) and 0xFF

        val luminance = (red * 0.2126f + green * 0.7152f + blue * 0.0722f) / 255

        return luminance.toDouble()
    }

    private fun activateNextLayer(
        currentLayer: Array<Double>,
        currentLayerWeights: Array<Array<Double>>,
        currentLayerOffsets: Array<Double>,
        nextLayer: Array<Double>
    ) {

        val vectorOfSums = getVectorOfPreActivationValues(
            currentLayer,
            currentLayerWeights,
            currentLayerOffsets
        )

        nextLayer.forEachIndexed { index, _ ->
            val currentSumValue = vectorOfSums[index]

            nextLayer[index] = activationFun(currentSumValue)
        }
    }

    fun activationFun(x: Double) = 1 / (1 + Math.exp(-x)) // сигмойда

    fun gradientFun(x: Double) = Math.exp(-x) / ((1 + Math.exp(-x)).pow(2)) // производная сигмойды

    private fun getVectorOfPreActivationValues(
        layer: Array<Double>,
        layerWeights: Array<Array<Double>>,
        layerOffsets: Array<Double>
    ): Array<Double> {
        val vectorOfSums = mutableListOf<Double>()

        layerWeights.forEachIndexed { index, row ->
            var weightedSum = 0.0

            row.forEachIndexed { index, weight ->
                val neuronValue = layer[index]

                weightedSum += neuronValue * weight
            }

            vectorOfSums.add(weightedSum + layerOffsets[index]) //Добавлений будет как высота матрицы весов что равно высоте следующего слоя
        }

        return vectorOfSums.toTypedArray()
    }

    private fun getResultNumber() = outputLayer.indexOf(outputLayer.max())

    /**
     * Learning
     */
    fun learning(image: BufferedImage, expectedResult: Int) {
        recognize(image)
        val expectedOutputs =
            arrayOf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0).apply { this[expectedResult] = 1.0 }

        val learningCondition = outputLayer.toList().filter {
            if (it == outputLayer.max()) {
                false
            } else {
                true
            }
        }.none {
            outputLayer.max() / 100 < it
        } && outputLayer.max() > 0.8

        if (!learningCondition) {
            calculateBackwardPass(expectedOutputs)
        }
    }

    private fun calculateBackwardPass(expectedOutputs: Array<Double>) {
        val vectorOfPreActivationValues = getVectorOfPreActivationValues(
            // высота вектора = высоте матрицы весов переданых для расчета
            layer = secondLayer,
            layerWeights = secondLayerWeightsValues,
            layerOffsets = secondLayerOffsetsValues,
        )

        val gradientOut = mutableListOf<Double>()

        outputLayer.forEachIndexed { index, output ->

            val derivativeErrorValue = 2 * (outputLayer[index] - expectedOutputs[index])

            val preActivationValue = vectorOfPreActivationValues[index]
            val derivativeOfActivation = gradientFun(preActivationValue)

            val gradientValue = derivativeErrorValue * derivativeOfActivation

            gradientOut.add(gradientValue)
        }

        val derivativeActivationVectorH2 = getVectorOfPreActivationValues(
            layer = firstLayer,
            layerWeights = firstLayerWeightsValues,
            layerOffsets = firstLayerOffsetsValues,
        ).map { gradientFun(it) }

        val gradientH2 =
            matrixMultiplication(secondLayerWeightsValues.transposition(), gradientOut).vectorWiseMultiplication(
                derivativeActivationVectorH2
            )

        val derivativeActivationVectorH1 = getVectorOfPreActivationValues(
            layer = inputLayerValues,
            layerWeights = inputWeightsValues,
            layerOffsets = inputOffsetsValues,
        ).map { gradientFun(it) }

        val gradientH1 =
            matrixMultiplication(firstLayerWeightsValues.transposition(), gradientH2).vectorWiseMultiplication(
                derivativeActivationVectorH1
            )

        //Составляем градиенты смещений и весов

        val (gradientVectorSecondOutputWeights, gradientVectorSecondOutputOffsets) = getWeightAndOffsetsGradients(
            layerGradient = gradientOut,
            nextLayer = secondLayer,
        )

        val (gradientVectorFirstSecondWeights, gradientVectorFirstSecondOffsets) = getWeightAndOffsetsGradients(
            layerGradient = gradientH2,
            nextLayer = firstLayer,
        )

        val (gradientVectorInputFirstWeights, gradientVectorInputFirstOffsets) = getWeightAndOffsetsGradients(
            layerGradient = gradientH1,
            nextLayer = inputLayerValues,
        )

        // Складываем градиенты в общую кучу

        addToGradient2(
            overallGradient = _gradientVectorSOWeights,
            exampleGradient = gradientVectorSecondOutputWeights
        )

        addToGradient(
            overallGradient = _gradientVectorSOOffsets,
            exampleGradient = gradientVectorSecondOutputOffsets
        )

        addToGradient2(
            overallGradient = _gradientVectorFSWeights,
            exampleGradient = gradientVectorFirstSecondWeights
        )

        addToGradient(
            overallGradient = _gradientVectorFSOffsets,
            exampleGradient = gradientVectorFirstSecondOffsets
        )

        addToGradient2(
            overallGradient = _gradientVectorIFWeights,
            exampleGradient = gradientVectorInputFirstWeights
        )

        addToGradient(
            overallGradient = _gradientVectorIFOffsets,
            exampleGradient = gradientVectorInputFirstOffsets
        )
    }

    /**
     * Расчет градиента на основе предыдущего
     */
    private fun getWeightAndOffsetsGradients(
        layerGradient: List<Double>,
        nextLayer: Array<Double>,
    ): Pair<List<List<Double>>, List<Double>> {
        val weightsGradient = mutableListOf<MutableList<Double>>()

        for (h in layerGradient.indices) {
            val row = mutableListOf<Double>()

            for (w in nextLayer.indices) {
                row.add(layerGradient[h] * nextLayer[w])
            }

            weightsGradient.add(row)
        }

        return weightsGradient to layerGradient
    }

    /**
     * Добавление в общий градиент
     */

    fun addToGradient(
        overallGradient: MutableList<Double>,
        exampleGradient: List<Double>
    ) {
        if (overallGradient.isEmpty()) {
            overallGradient.addAll(exampleGradient)
        } else {
            overallGradient.mapIndexed { index, value ->
                if (exampleGradient[index] == 0.0) {
                    value
                } else {
                    value + exampleGradient[index]
                }
            }
        }
    }

    fun addToGradient2(
        overallGradient: MutableList<MutableList<Double>>,
        exampleGradient: List<List<Double>>
    ) {
        if (overallGradient.isEmpty()) {
            exampleGradient.forEach { overallGradient.add(it.toMutableList()) }
        } else {
            overallGradient.mapIndexed { h, row ->
                row.mapIndexed { w, value ->
                    exampleGradient[h][w] + value
                }
            }
        }
    }

    fun applyAllGradients() {

        applyGradient(
            layerWeights = secondLayerWeightsValues,
            layerOffsets = secondLayerOffsetsValues,
            gradientOfWeights = _gradientVectorSOWeights,
            gradientOfOffsets = _gradientVectorSOOffsets.toTypedArray()
        )

        applyGradient(
            layerWeights = firstLayerWeightsValues,
            layerOffsets = firstLayerOffsetsValues,
            gradientOfWeights = _gradientVectorFSWeights,
            gradientOfOffsets = _gradientVectorFSOffsets.toTypedArray()
        )

        applyGradient(
            layerWeights = inputWeightsValues,
            layerOffsets = inputOffsetsValues,
            gradientOfWeights = _gradientVectorIFWeights,
            gradientOfOffsets = _gradientVectorIFOffsets.toTypedArray()
        )

        _gradientVectorSOWeights.clear()
        _gradientVectorSOOffsets.clear()
        _gradientVectorFSWeights.clear()
        _gradientVectorFSOffsets.clear()
        _gradientVectorIFWeights.clear()
        _gradientVectorIFOffsets.clear()
    }

    /**
     * Изменение весов на основе градиента
     */
    private fun applyGradient(
        layerWeights: Array<Array<Double>>,
        layerOffsets: Array<Double>,
        gradientOfWeights: List<List<Double>>,
        gradientOfOffsets: Array<Double>,
    ) {
        // Делим на 100 потому что столько было сложено градиентов
        layerWeights.forEachIndexed { heightIndex, row ->

            row.forEachIndexed { wightIndex, weight ->
                row[wightIndex] =
                    row[wightIndex] + LEARNING_SPEED * (-gradientOfWeights[heightIndex][wightIndex] / COUNT_SUMMED_GRADIENTS) // Изменение веса в соответсвии с градиентом
            }
        }

        layerOffsets.forEachIndexed { index, offset ->
            layerOffsets[index] =
                layerOffsets[index] + LEARNING_SPEED * (-gradientOfOffsets[index] / COUNT_SUMMED_GRADIENTS) // Изменение смещения в соответсвии с градиентом
        }
    }


    /**
     * Для тестирования
     **/
    fun recognize(number: Int): Int {
        inputLayerValues[0] = number.toDouble()

        return activatingLayers()
    }

    fun learning(number: Int, expectedResult: Int) {
        recognize(number)
        val expectedOutputs = arrayOf(0.0, 0.0).apply { this[expectedResult] = 1.0 }

        calculateBackwardPass(expectedOutputs)
    }

    fun recognize(left: Int, right: Int): Int {
        inputLayerValues[0] = left.toDouble()
        inputLayerValues[1] = right.toDouble()

        return activatingLayers()
    }

    fun learning(left: Int, right: Int, expectedResult: Int) {
        recognize(left, right)
        val expectedOutputs = arrayOf(0.0, 0.0).apply { this[expectedResult] = 1.0 }

        calculateBackwardPass(expectedOutputs)
    }
}