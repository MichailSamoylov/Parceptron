package parseptron

import java.awt.image.BufferedImage
import kotlin.math.absoluteValue
import kotlin.math.max
import kotlin.math.pow
import kotlin.random.Random

class Parseptron {

    companion object {

        const val INPUTS_SIZE = 784 // кол-во пикселей на картинке
        const val FIRST_LAYER_SIZE = 10
        const val SECOND_LAYER_SIZE = 10
        const val OUTPUT_SIZE = 10

        const val LEARNING_SPEED = 0.1f

        val VALUE_RANGE = 0..10
    }

    private val gradientVectorSOWeights = mutableListOf<Float>()
    private val gradientVectorFSWeights = mutableListOf<Float>()
    private val gradientVectorIFWeights = mutableListOf<Float>()
    private val gradientVectorSOOffsets = mutableListOf<Float>()
    private val gradientVectorFSOffsets = mutableListOf<Float>()
    private val gradientVectorIFOffsets = mutableListOf<Float>()

    val inputLayerValues: Array<Float> = getRandomVector(INPUTS_SIZE)
    val inputWeightsValues: Array<Array<Float>> = getRandomMatrix(width = INPUTS_SIZE, height = FIRST_LAYER_SIZE)
    private val inputOffsetsValues: Array<Float> = getRandomVector(height = FIRST_LAYER_SIZE)

    val firstLayer: Array<Float> = getRandomVector(FIRST_LAYER_SIZE)
    val firstLayerWeightsValues: Array<Array<Float>> =
        getRandomMatrix(width = FIRST_LAYER_SIZE, height = SECOND_LAYER_SIZE)
    private val firstLayerOffsetsValues: Array<Float> = getRandomVector(height = SECOND_LAYER_SIZE)

    val secondLayer: Array<Float> = getRandomVector(SECOND_LAYER_SIZE)
    val secondLayerWeightsValues: Array<Array<Float>> = getRandomMatrix(width = SECOND_LAYER_SIZE, height = OUTPUT_SIZE)
    private val secondLayerOffsetsValues: Array<Float> = getRandomVector(height = OUTPUT_SIZE)

    val outputLayer: Array<Float> = getRandomVector(OUTPUT_SIZE)

    private fun getRandomVector(height: Int): Array<Float> {
        val vector = mutableListOf<Float>()

        for (j in 0 until height) {
            val value = getRandom() / 10.0

            vector.add(value.toFloat())
        }

        return vector.toTypedArray()
    }

    private fun getRandomMatrix(width: Int, height: Int): Array<Array<Float>> {
        val matrix = mutableListOf<Array<Float>>()

        for (i in 0 until height) {
            val row = mutableListOf<Float>()
            for (j in 0 until width) {
                val weight = getRandom() / 10.0

                row.add(weight.toFloat())
            }
            matrix.add(row.toTypedArray())
        }

        return matrix.toTypedArray()
    }

    private fun getRandom() = Random.nextInt(10)

    fun recognize(image: BufferedImage): Int {
        fillInputLayer(image)

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

    fun recognize(number: Int): Int {
        inputLayerValues[0] = number.toFloat()

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

    private fun getPixelValue(color: Int): Float {
        val red = (color ushr 16) and 0xFF
        val green = (color ushr 8) and 0xFF
        val blue = (color ushr 0) and 0xFF

        val luminance = (red * 0.2126f + green * 0.7152f + blue * 0.0722f) / 255

        return luminance
    }

    private fun activateNextLayer(
        currentLayer: Array<Float>,
        currentLayerWeights: Array<Array<Float>>,
        currentLayerOffsets: Array<Float>,
        nextLayer: Array<Float>
    ) {

        val vectorOfSums = getVectorOfPreActivationValues(
            currentLayer,
            currentLayerWeights,
            currentLayerOffsets
        )

        nextLayer.forEachIndexed { index, _ ->
            val currentSumValue = vectorOfSums[index].toDouble()

            nextLayer[index] = activationFun(currentSumValue).toFloat()
        }
    }

    private fun activationFun(x: Double) = (1 / (1 + Math.E.pow(-x))) // сигмойда

    private fun gradientFun(x: Double) = Math.E.pow(-x) / ((1 + Math.E.pow(-x)).pow(2)) // производная сигмойды

    private fun getVectorOfPreActivationValues(
        layer: Array<Float>,
        layerWeights: Array<Array<Float>>,
        layerOffsets: Array<Float>
    ): Array<Float> {
        val vectorOfSums = mutableListOf<Float>()

        layerWeights.forEachIndexed { index, row ->
            var weightedSum = 0f

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
        val expectedOutputs = arrayOf(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0).apply { this[expectedResult] = 1.0 }

        //Расчитать градиент для весов между вторым и выходм слоем
        //Расчитать новые веса между вторым и выходным слоями

        val vectorOfPreActivationValues = getVectorOfPreActivationValues(
            // высота вектора = высоте матрицы весов переданых для расчета
            layer = secondLayer,
            layerWeights = secondLayerWeightsValues,
            layerOffsets = secondLayerOffsetsValues,
        )

        val gradientVectorSecondOutputWeights = mutableListOf<Float>()

        secondLayerWeightsValues.forEachIndexed { heightIndex, row ->
            row.forEachIndexed { wightIndex, weight ->

                val derivativeErrorValue = 2 * (outputLayer[heightIndex] - expectedOutputs[heightIndex])

                val tiedNeuron = secondLayer[wightIndex] // wightIndex - размерность предыдущего слоя

                val preActivationValue = vectorOfPreActivationValues[heightIndex]
                val derivativeOfActivation =
                    gradientFun(preActivationValue.toDouble()) //heightIndex используется для взятия значений из векторов размером выходного слоя

                val gradientValue = derivativeErrorValue * tiedNeuron * derivativeOfActivation

                gradientVectorSecondOutputWeights.add(gradientValue.toFloat())
            }
        }

        val gradientVectorSecondOutputOffsets = mutableListOf<Float>()

        secondLayerOffsetsValues.forEachIndexed { heightIndex, offset -> // heightIndex - высота вектора смещений совпадает с высотой вектора значений нейронов в след слое
            val derivativeErrorValue = 2 * (outputLayer[heightIndex] - expectedOutputs[heightIndex])
            val derivativeOfActivation = gradientFun(vectorOfPreActivationValues[heightIndex].toDouble())

            val gradientValue = derivativeErrorValue * derivativeOfActivation //Значение dZ / db = 1

            gradientVectorSecondOutputOffsets.add(gradientValue.toFloat())
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
            overallGradient = gradientVectorSOWeights,
            exampleGradient = gradientVectorSecondOutputWeights
        )

        addToGradient(
            overallGradient = gradientVectorSOOffsets,
            exampleGradient = gradientVectorSecondOutputOffsets
        )

        addToGradient(
            overallGradient = gradientVectorFSWeights,
            exampleGradient = gradientVectorFirstSecondWeights
        )

        addToGradient(
            overallGradient = gradientVectorFSOffsets,
            exampleGradient = gradientVectorFirstSecondOffsets
        )

        addToGradient(
            overallGradient = gradientVectorIFWeights,
            exampleGradient = gradientVectorInputFirstWeights
        )

        addToGradient(
            overallGradient = gradientVectorIFOffsets,
            exampleGradient = gradientVectorInputFirstOffsets
        )
    }

    fun addToGradient(
        overallGradient: MutableList<Float>,
        exampleGradient: List<Float>
    ) {
        if (overallGradient.isEmpty()) {
            overallGradient.addAll(exampleGradient)
        } else {
            //Усреднить или сложить ?
            overallGradient.mapIndexed { index, value ->
                (value + exampleGradient[index]) / 2
            }
        }
    }

    fun applyAllGradients() {

        applyGradient(
            layerWeights = secondLayerWeightsValues,
            layerOffsets = secondLayerOffsetsValues,
            gradientOfWeights = gradientVectorSOWeights.toTypedArray(),
            gradientOfOffsets = gradientVectorSOOffsets.toTypedArray()
        )

        applyGradient(
            layerWeights = firstLayerWeightsValues,
            layerOffsets = firstLayerOffsetsValues,
            gradientOfWeights = gradientVectorFSWeights.toTypedArray(),
            gradientOfOffsets = gradientVectorFSOffsets.toTypedArray()
        )

        applyGradient(
            layerWeights = inputWeightsValues,
            layerOffsets = inputOffsetsValues,
            gradientOfWeights = gradientVectorIFWeights.toTypedArray(),
            gradientOfOffsets = gradientVectorIFOffsets.toTypedArray()
        )

        gradientVectorSOWeights.clear()
        gradientVectorSOOffsets.clear()
        gradientVectorFSWeights.clear()
        gradientVectorFSOffsets.clear()
        gradientVectorIFWeights.clear()
        gradientVectorIFOffsets.clear()
    }

    private fun getWeightAndOffsetsGradients(
        layer: Array<Float>,
        layerWeights: Array<Array<Float>>,
        layerOffsets: Array<Float>,
        nextLayer: Array<Float>,
        nextLayerWeights: Array<Array<Float>>,
        gradientOfNextWeightsLayer: Array<Float>
    ): Pair<List<Float>, List<Float>> {

        // Расчет градиентов для первого слоя
        val vectorOfPreActivationValues = getVectorOfPreActivationValues(
            // высота вектора = высоте матрицы весов переданых для расчета
            layer = layer,
            layerWeights = layerWeights,
            layerOffsets = layerOffsets,
        )

        val gradientVectorCurrentNextWeights = mutableListOf<Float>()

        layerWeights.forEachIndexed { heightIndex, row ->
            row.forEachIndexed { wightIndex, weight ->

                var dC_By_daprev = 0f

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
                    gradientFun(vectorOfPreActivationValues[heightIndex].toDouble()) //heightIndex используется для взятия значений из векторов размером выходного слоя
                val tiedNeuron = layer[wightIndex] // an-2

                val gradientValue = dC_By_daprev * tiedNeuron * derivativeOfActivation

                gradientVectorCurrentNextWeights.add(gradientValue.toFloat())
            }
        }

        val gradientVectorCurrentNextOffsets = mutableListOf<Float>()

        layerOffsets.forEachIndexed { heightIndex, offset ->

            var dC_By_daprev = 0f

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
                gradientFun(vectorOfPreActivationValues[heightIndex].toDouble()) //heightIndex используется для взятия значений из векторов размером выходного слоя

            val gradientValue = dC_By_daprev * derivativeOfActivation

            gradientVectorCurrentNextOffsets.add(gradientValue.toFloat())
        }


        return gradientVectorCurrentNextWeights to gradientVectorCurrentNextOffsets
    }

    private fun applyGradient(
        layerWeights: Array<Array<Float>>,
        layerOffsets: Array<Float>,
        gradientOfWeights: Array<Float>,
        gradientOfOffsets: Array<Float>,
    ) {
        var gradientCounter = 0

        layerWeights.forEachIndexed { heightIndex, row ->

            row.forEachIndexed { wightIndex, weight ->
                row[wightIndex] =
                    row[wightIndex] + LEARNING_SPEED * (-gradientOfWeights[gradientCounter]) //Изменение веса в соответсвии с градиентом

                gradientCounter++
            }
        }

        layerOffsets.forEachIndexed { index, offset ->
            layerOffsets[index] =
                layerOffsets[index] + LEARNING_SPEED * (-gradientOfOffsets[index]) //Изменение смещения в соответсвии с градиентом
        }
    }

//    private fun applyGradient(
//        layerWeights: Array<Array<Float>>,
//        layerOffsets: Array<Float>,
//        gradientOfWeights: Array<Float>,
//        gradientOfOffsets: Array<Float>,
//    ) {
//        var gradientCounter = 0
//
//        layerWeights.forEachIndexed { heightIndex, row ->
//
//            row.forEachIndexed { wightIndex, weight ->
//                val currentGradientValue = (-gradientOfWeights[gradientCounter])
//                var currentLearningStep = LEARNING_SPEED
//                while (currentLearningStep > currentGradientValue.absoluteValue) {
//                    currentLearningStep /= 10
//                }
//
//                row[wightIndex] =
//                    row[wightIndex] + currentLearningStep * (-gradientOfWeights[gradientCounter]) //Изменение веса в соответсвии с градиентом
//
//                gradientCounter++
//            }
//        }
//
//        layerOffsets.forEachIndexed { index, offset ->
//            val currentGradientValue = (-gradientOfOffsets[index])
//            var currentLearningStep = LEARNING_SPEED
//            while (currentLearningStep > currentGradientValue.absoluteValue) {
//                currentLearningStep /= 10
//            }
//
//            layerOffsets[index] =
//                layerOffsets[index] + currentLearningStep * (-gradientOfOffsets[index]) //Изменение смещения в соответсвии с градиентом
//        }
//    }
//    fun learning(number: Int, expectedResult: Int) {
//        val actualResult = recognize(number)
//        val expectedOutputs = arrayOf(0.0, 0.0).apply { this[expectedResult] = 1.0 }
//
//        println("Значение ошибки 1: ${(outputLayer[0] - expectedOutputs[0]).pow(2)}")
//        println("Значение ошибки 2: ${(outputLayer[1] - expectedOutputs[1]).pow(2)}")
//
//        //Расчитать градиент для весов между вторым и выходм слоем
//        //Расчитать новые веса между вторым и выходным слоями
//
//        val vectorOfPreActivationValues = getVectorOfPreActivationValues(
//            // высота вектора = высоте матрицы весов переданых для расчета
//            layer = secondLayer,
//            layerWeights = secondLayerWeightsValues,
//            layerOffsets = secondLayerOffsetsValues,
//        )
//
//        val gradientVectorSecondOutputWeights = mutableListOf<Float>()
//
//        secondLayerWeightsValues.forEachIndexed { heightIndex, row ->
//            row.forEachIndexed { wightIndex, weight ->
//
//                val derivativeErrorValue = 2 * (outputLayer[heightIndex] - expectedOutputs[heightIndex])
//
//                val tiedNeuron = secondLayer[wightIndex] // wightIndex - размерность предыдущего слоя
//
//                val preActivationValue = vectorOfPreActivationValues[heightIndex]
//                val derivativeOfActivation =
//                    gradientFun(preActivationValue.toDouble()) //heightIndex используется для взятия значений из векторов размером выходного слоя
//
//                val gradientValue = derivativeErrorValue * tiedNeuron * derivativeOfActivation
//
//                gradientVectorSecondOutputWeights.add(gradientValue.toFloat())
//            }
//        }
//
//        val gradientVectorSecondOutputOffsets = mutableListOf<Float>()
//
//        secondLayerOffsetsValues.forEachIndexed { heightIndex, offset -> // heightIndex - высота вектора смещений совпадает с высотой вектора значений нейронов в след слое
//            val derivativeErrorValue = 2 * (outputLayer[heightIndex] - expectedOutputs[heightIndex])
//            val derivativeOfActivation = gradientFun(vectorOfPreActivationValues[heightIndex].toDouble())
//
//            val gradientValue = derivativeErrorValue * derivativeOfActivation //Значение dZ / db = 1
//
//            gradientVectorSecondOutputOffsets.add(gradientValue.toFloat())
//        }
//
//        val (gradientVectorFirstSecondWeights, gradientVectorFirstSecondOffsets) = getWeightAndOffsetsGradients(
//            layer = firstLayer,
//            layerWeights = firstLayerWeightsValues,
//            layerOffsets = firstLayerOffsetsValues,
//            nextLayer = secondLayer,
//            nextLayerWeights = secondLayerWeightsValues,
//            gradientOfNextWeightsLayer = gradientVectorSecondOutputWeights.toTypedArray(),
//        )
//
//        val (gradientVectorInputFirstWeights, gradientVectorInputFirstOffsets) = getWeightAndOffsetsGradients(
//            layer = inputLayerValues,
//            layerWeights = inputWeightsValues,
//            layerOffsets = inputOffsetsValues,
//            nextLayer = firstLayer,
//            nextLayerWeights = firstLayerWeightsValues,
//            gradientOfNextWeightsLayer = gradientVectorFirstSecondWeights.toTypedArray(),
//        )
//
//        applyGradient(
//            layerWeights = secondLayerWeightsValues,
//            layerOffsets = secondLayerOffsetsValues,
//            gradientOfWeights = gradientVectorSecondOutputWeights.toTypedArray(),
//            gradientOfOffsets = gradientVectorSecondOutputOffsets.toTypedArray()
//        )
//
//        applyGradient(
//            layerWeights = firstLayerWeightsValues,
//            layerOffsets = firstLayerOffsetsValues,
//            gradientOfWeights = gradientVectorFirstSecondWeights.toTypedArray(),
//            gradientOfOffsets = gradientVectorFirstSecondOffsets.toTypedArray()
//        )
//
//        applyGradient(
//            layerWeights = inputWeightsValues,
//            layerOffsets = inputOffsetsValues,
//            gradientOfWeights = gradientVectorInputFirstWeights.toTypedArray(),
//            gradientOfOffsets = gradientVectorInputFirstOffsets.toTypedArray()
//        )
//    }
}