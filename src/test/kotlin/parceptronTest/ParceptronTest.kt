package parceptronTest

import org.junit.Assert
import parseptron.Parseptron
import kotlin.test.Test

class ParceptronTest {

    @Test
    fun `learning EXPECT check output to second weights layer gradient calculation`() {
        val expectedOutputIndex = 1
        val expectedOutputs = arrayOf(0f, 1f)

        val parceptron = Parseptron(
            inputSize = 1,
            firstLayerSize = 3,
            secondLayerSize = 2,
            outputSize = 2
        )

        parceptron.learning(1, expectedOutputIndex)

        val derivativeErrorValue1 = 2 * (parceptron.outputLayer[0] - expectedOutputs[0])
        val derivativeErrorValue2 = 2 * (parceptron.outputLayer[1] - expectedOutputs[1])

        val preSum1 = parceptron.secondLayer[0] * parceptron.secondLayerWeightsValues[0][0] + parceptron.secondLayer[1] * parceptron.secondLayerWeightsValues[0][1] + parceptron.secondLayerOffsetsValues[0]
        val preSum2 = parceptron.secondLayer[0] * parceptron.secondLayerWeightsValues[1][0] + parceptron.secondLayer[1] * parceptron.secondLayerWeightsValues[1][1] + parceptron.secondLayerOffsetsValues[1]

        val expectedGradient = mutableListOf(
            (derivativeErrorValue1 * parceptron.gradientFun(preSum1.toDouble()) * parceptron.secondLayer[0]).toFloat(),
            (derivativeErrorValue1 * parceptron.gradientFun(preSum1.toDouble()) * parceptron.secondLayer[1]).toFloat(),
            (derivativeErrorValue2 * parceptron.gradientFun(preSum2.toDouble()) * parceptron.secondLayer[0]).toFloat(),
            (derivativeErrorValue2 * parceptron.gradientFun(preSum2.toDouble()) * parceptron.secondLayer[1]).toFloat(),
        )

        Assert.assertArrayEquals(expectedGradient.toTypedArray(), parceptron._gradientVectorSOWeights.toTypedArray())
    }

    @Test
    fun `learning EXPECT check output to second offsets layer gradient calculation`() {
        val expectedOutputIndex = 1
        val expectedOutputs = arrayOf(0f, 1f)

        val parceptron = Parseptron(
            inputSize = 1,
            firstLayerSize = 3,
            secondLayerSize = 2,
            outputSize = 2
        )

        parceptron.learning(1, expectedOutputIndex)

        val derivativeErrorValue1 = 2 * (parceptron.outputLayer[0] - expectedOutputs[0])
        val derivativeErrorValue2 = 2 * (parceptron.outputLayer[1] - expectedOutputs[1])

        val preSum1 = parceptron.secondLayer[0] * parceptron.secondLayerWeightsValues[0][0] + parceptron.secondLayer[1] * parceptron.secondLayerWeightsValues[0][1] + parceptron.secondLayerOffsetsValues[0]
        val preSum2 = parceptron.secondLayer[0] * parceptron.secondLayerWeightsValues[1][0] + parceptron.secondLayer[1] * parceptron.secondLayerWeightsValues[1][1] + parceptron.secondLayerOffsetsValues[1]

        val expectedGradient = mutableListOf(
            (derivativeErrorValue1 * parceptron.gradientFun(preSum1.toDouble())).toFloat(),
            (derivativeErrorValue2 * parceptron.gradientFun(preSum2.toDouble())).toFloat(),
        )

        Assert.assertArrayEquals(expectedGradient.toTypedArray(), parceptron._gradientVectorSOOffsets.toTypedArray())
    }

    @Test
    fun `learning EXPECT check second to first weights layer gradient calculation from previous gradient`() {
        val expectedOutputIndex = 1

        val parceptron = Parseptron(
            inputSize = 1,
            firstLayerSize = 3,
            secondLayerSize = 2,
            outputSize = 2
        )

        parceptron.learning(1, expectedOutputIndex)

        val derivativeErrorValue1 = parceptron._gradientVectorSOWeights[0]
        val derivativeErrorValue2 = parceptron._gradientVectorSOWeights[1]

        val preSum1 = parceptron.secondLayer[0] * parceptron.secondLayerWeightsValues[0][0] + parceptron.secondLayer[1] * parceptron.secondLayerWeightsValues[0][1] + parceptron.secondLayerOffsetsValues[0]
        val preSum2 = parceptron.secondLayer[0] * parceptron.secondLayerWeightsValues[1][0] + parceptron.secondLayer[1] * parceptron.secondLayerWeightsValues[1][1] + parceptron.secondLayerOffsetsValues[1]



        val expectedGradient = mutableListOf(
            (derivativeErrorValue1 * parceptron.gradientFun(preSum1.toDouble())).toFloat(),
            (derivativeErrorValue2 * parceptron.gradientFun(preSum2.toDouble())).toFloat(),
        )

        Assert.assertArrayEquals(expectedGradient.toTypedArray(), parceptron._gradientVectorFSWeights.toTypedArray())
    }
}