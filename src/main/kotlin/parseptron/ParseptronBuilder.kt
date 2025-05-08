package parseptron

import java.awt.image.BufferedImage
import kotlin.math.absoluteValue
import kotlin.math.pow
import kotlin.random.Random

class Parseptron(
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

        const val LEARNING_SPEED = 1f

        val VALUE_RANGE = 0..10
    }

    val _gradientVectorSOWeights = mutableListOf<Double>()
    val _gradientVectorFSWeights = mutableListOf<Double>()
    val _gradientVectorIFWeights = mutableListOf<Double>()
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
            val value = getRandom() / 10.0

            vector.add(value)
        }

        return vector.toTypedArray()
    }

    private fun getRandomMatrix(width: Int, height: Int): Array<Array<Double>> {
        val matrix = mutableListOf<Array<Double>>()

        for (i in 0 until height) {
            val row = mutableListOf<Double>()
            for (j in 0 until width) {
                val weight = getRandom() / 10.0

                row.add(weight)
            }
            matrix.add(row.toTypedArray())
        }

        return matrix.toTypedArray()
    }

    private fun getRandom() = Random.nextInt(10)

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
                    inputLayerValues[inputCounter] = getPixelValue(color)
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
            val currentSumValue = vectorOfSums[index].toDouble()

            nextLayer[index] = activationFun(currentSumValue)
        }
    }

    fun activationFun(x: Double) = (1 / (1 + Math.E.pow(-x))) // сигмойда

    fun gradientFun(x: Double) = Math.E.pow(-x) / ((1 + Math.E.pow(-x)).pow(2)) // производная сигмойды

    private fun getVectorOfPreActivationValues(
        layer: Array<Double>,
        layerWeights: Array<Array<Double>>,
        layerOffsets: Array<Double>
    ): Array<Double> {
        val vectorOfSums = mutableListOf<Double>()

        layerWeights.forEachIndexed { index, row ->
            var weightedSum = 0.0

            row.forEachIndexed { index, weight ->
                val previousNeuronValue = layer[index]

                weightedSum += previousNeuronValue * weight
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

        calculateGradients(expectedOutputs)
    }

    private fun calculateGradients(expectedOutputs: Array<Double>) {
        //Расчитать градиент для весов между вторым и выходм слоем
        //Расчитать новые веса между вторым и выходным слоями

        val vectorOfPreActivationValues = getVectorOfPreActivationValues(
            // высота вектора = высоте матрицы весов переданых для расчета
            layer = secondLayer,
            layerWeights = secondLayerWeightsValues,
            layerOffsets = secondLayerOffsetsValues,
        )

        val gradientVectorSecondOutputWeights = mutableListOf<Double>()

        secondLayerWeightsValues.forEachIndexed { heightIndex, row ->
            row.forEachIndexed { wightIndex, weight ->

                val derivativeErrorValue = 2 * (outputLayer[heightIndex] - expectedOutputs[heightIndex])

                val tiedNeuron = secondLayer[wightIndex] // wightIndex - размерность предыдущего слоя

                val preActivationValue = vectorOfPreActivationValues[heightIndex]
                val derivativeOfActivation =
                    gradientFun(preActivationValue) //heightIndex используется для взятия значений из векторов размером выходного слоя

                val gradientValue = derivativeErrorValue * tiedNeuron * derivativeOfActivation

                gradientVectorSecondOutputWeights.add(gradientValue)
            }
        }

        val gradientVectorSecondOutputOffsets = mutableListOf<Double>()

        secondLayerOffsetsValues.forEachIndexed { heightIndex, offset -> // heightIndex - высота вектора смещений совпадает с высотой вектора значений нейронов в след слое
            val derivativeErrorValue = 2 * (outputLayer[heightIndex] - expectedOutputs[heightIndex])
            val derivativeOfActivation = gradientFun(vectorOfPreActivationValues[heightIndex])

            val gradientValue = derivativeErrorValue * derivativeOfActivation //Значение dZ / db = 1

            gradientVectorSecondOutputOffsets.add(gradientValue)
        }

        val (gradientVectorFirstSecondWeights, gradientVectorFirstSecondOffsets) = getWeightAndOffsetsGradients(
            layer = firstLayer,
            layerWeights = firstLayerWeightsValues,
            layerOffsets = firstLayerOffsetsValues,
            nextLayer = secondLayer,
            nextLayerWeights = secondLayerWeightsValues,
            gradientOfNextWeightsLayer = gradientVectorSecondOutputWeights.toTypedArray(),
        )

        val (gradientVectorInputFirstWeights, gradientVectorInputFirstOffsets) = getWeightAndOffsetsGradients(
            layer = inputLayerValues,
            layerWeights = inputWeightsValues,
            layerOffsets = inputOffsetsValues,
            nextLayer = firstLayer,
            nextLayerWeights = firstLayerWeightsValues,
            gradientOfNextWeightsLayer = gradientVectorFirstSecondWeights.toTypedArray(),
        )

        addToGradient(
            overallGradient = _gradientVectorSOWeights,
            exampleGradient = gradientVectorSecondOutputWeights
        )

        addToGradient(
            overallGradient = _gradientVectorSOOffsets,
            exampleGradient = gradientVectorSecondOutputOffsets
        )

        addToGradient(
            overallGradient = _gradientVectorFSWeights,
            exampleGradient = gradientVectorFirstSecondWeights
        )

        addToGradient(
            overallGradient = _gradientVectorFSOffsets,
            exampleGradient = gradientVectorFirstSecondOffsets
        )

        addToGradient(
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
        layer: Array<Double>,
        layerWeights: Array<Array<Double>>,
        layerOffsets: Array<Double>,
        nextLayer: Array<Double>,
        nextLayerWeights: Array<Array<Double>>,
        gradientOfNextWeightsLayer: Array<Double>
    ): Pair<List<Double>, List<Double>> {

        // Расчет градиентов для первого слоя
        val vectorOfPreActivationValues = getVectorOfPreActivationValues(
            // высота вектора = высоте матрицы весов переданых для расчета
            layer = layer,
            layerWeights = layerWeights,
            layerOffsets = layerOffsets,
        )

        val gradientVectorCurrentNextWeights = mutableListOf<Double>()

        layerWeights.forEachIndexed { heightIndex, row ->
            row.forEachIndexed { wightIndex, weight ->

                //Testing val stringBuilder = StringBuilder()

                var dC_By_daprev = 0.0

                //Указатель на индекс для расчитаного значения градиента в gradientVectorSecondOutputWeights, расчитанного для secondLayer[wightIndex]
                val gradientMatrix = mutableListOf<MutableList<Double>>()
                var gradientVectorCounter = 0

                for (h in nextLayerWeights.indices) {
                    val line = mutableListOf<Double>()

                    for (w in 0 until nextLayerWeights[0].size) {

                        line.add(gradientOfNextWeightsLayer[gradientVectorCounter])
                        gradientVectorCounter++
                    }

                    gradientMatrix.add(line)
                }

                // Смотреть появснение п1
                for (h in nextLayerWeights.indices) {

                    val tiedNeuron = nextLayer[heightIndex]
                    val prevWeight = nextLayerWeights[h][heightIndex]

                    dC_By_daprev += gradientMatrix[h][heightIndex] / tiedNeuron * prevWeight

                    //stringBuilder.append("${gradientMatrix[h][heightIndex]} / $tiedNeuron * $prevWeight +")
                }

                val derivativeOfActivation =
                    gradientFun(vectorOfPreActivationValues[heightIndex]) //heightIndex используется для взятия значений из векторов размером выходного слоя
                val tiedNeuron = layer[wightIndex] // an-2

                //println("gradientSum $dC_By_daprev = $stringBuilder")

                val gradientValue = dC_By_daprev * tiedNeuron * derivativeOfActivation

                gradientVectorCurrentNextWeights.add(gradientValue)
            }
        }

        val gradientVectorCurrentNextOffsets = mutableListOf<Double>()

        layerOffsets.forEachIndexed { heightIndex, offset ->

            var dC_By_daprev = 0.0

            //Указатель на индекс для расчитаного значения градиента в gradientVectorSecondOutputWeights, расчитанного для secondLayer[wightIndex]
            var counter = 0

            nextLayerWeights.forEachIndexed { _, row ->
                val tiedNeuron = nextLayer[heightIndex]
                val prevWeight = row[heightIndex]

                row.forEachIndexed { index, pw ->

                    //TODO Если не будет работать то грешить на это место
                    if (pw == prevWeight) { // Эта операция нужна чтобы исключить из суммы произвдные ошибки по нейронам не связанным с tiedNeuron
                        dC_By_daprev += gradientOfNextWeightsLayer[counter] / tiedNeuron * prevWeight
                    }

                    counter++
                }
            }

            val derivativeOfActivation =
                gradientFun(vectorOfPreActivationValues[heightIndex]) //heightIndex используется для взятия значений из векторов размером выходного слоя

            val gradientValue = dC_By_daprev * derivativeOfActivation

            gradientVectorCurrentNextOffsets.add(gradientValue)
        }

        return gradientVectorCurrentNextWeights to gradientVectorCurrentNextOffsets
    }

    /**
     * Добавление в общий градиент
     */
    fun addToGradient(
        overallGradient: MutableList<Double>,
        exampleGradient: List<Double>
    ) {
        val max = exampleGradient.max()
        val min = exampleGradient.min()
        val border = max - ((max - min) / 4 * 1)
        val filtredExampleGradient = exampleGradient//.map { if (it.absoluteValue < border.absoluteValue) 0f else it }

        if (overallGradient.isEmpty()) {
            overallGradient.addAll(filtredExampleGradient)
        } else {
            overallGradient.mapIndexed { index, value ->
                if (filtredExampleGradient[index] == 0.0) {
                    value
                } else {
                    (value + filtredExampleGradient[index]) / 2
                }
            }
        }
    }

    fun applyAllGradients() {

        applyGradient(
            layerWeights = secondLayerWeightsValues,
            layerOffsets = secondLayerOffsetsValues,
            gradientOfWeights = _gradientVectorSOWeights.toTypedArray(),
            gradientOfOffsets = _gradientVectorSOOffsets.toTypedArray()
        )

        applyGradient(
            layerWeights = firstLayerWeightsValues,
            layerOffsets = firstLayerOffsetsValues,
            gradientOfWeights = _gradientVectorFSWeights.toTypedArray(),
            gradientOfOffsets = _gradientVectorFSOffsets.toTypedArray()
        )

        applyGradient(
            layerWeights = inputWeightsValues,
            layerOffsets = inputOffsetsValues,
            gradientOfWeights = _gradientVectorIFWeights.toTypedArray(),
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
        gradientOfWeights: Array<Double>,
        gradientOfOffsets: Array<Double>,
    ) {
        var gradientCounter = 0

        layerWeights.forEachIndexed { heightIndex, row ->

            row.forEachIndexed { wightIndex, weight ->
                row[wightIndex] =
                    row[wightIndex] + LEARNING_SPEED*(-gradientOfWeights[gradientCounter]) // Изменение веса в соответсвии с градиентом
            }
            gradientCounter++
        }

        layerOffsets.forEachIndexed { index, offset ->
            layerOffsets[index] =
                layerOffsets[index] + LEARNING_SPEED*(-gradientOfOffsets[index]) // Изменение смещения в соответсвии с градиентом
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

        calculateGradients(expectedOutputs)
    }
}