package parceptronTest

import org.junit.Assert
import parseptron.Parseptron
import java.math.BigDecimal
import java.math.RoundingMode
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

        val preSum1 =
            parceptron.secondLayer[0] * parceptron.secondLayerWeightsValues[0][0] + parceptron.secondLayer[1] * parceptron.secondLayerWeightsValues[0][1] + parceptron.secondLayerOffsetsValues[0]
        val preSum2 =
            parceptron.secondLayer[0] * parceptron.secondLayerWeightsValues[1][0] + parceptron.secondLayer[1] * parceptron.secondLayerWeightsValues[1][1] + parceptron.secondLayerOffsetsValues[1]

        val expectedGradient = mutableListOf(
            (derivativeErrorValue1 * parceptron.gradientFun(preSum1) * parceptron.secondLayer[0]),
            (derivativeErrorValue1 * parceptron.gradientFun(preSum1) * parceptron.secondLayer[1]),
            (derivativeErrorValue2 * parceptron.gradientFun(preSum2) * parceptron.secondLayer[0]),
            (derivativeErrorValue2 * parceptron.gradientFun(preSum2) * parceptron.secondLayer[1]),
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

        val preSum1 =
            parceptron.secondLayer[0] * parceptron.secondLayerWeightsValues[0][0] + parceptron.secondLayer[1] * parceptron.secondLayerWeightsValues[0][1] + parceptron.secondLayerOffsetsValues[0]
        val preSum2 =
            parceptron.secondLayer[0] * parceptron.secondLayerWeightsValues[1][0] + parceptron.secondLayer[1] * parceptron.secondLayerWeightsValues[1][1] + parceptron.secondLayerOffsetsValues[1]

        val expectedGradient = mutableListOf(
            (derivativeErrorValue1 * parceptron.gradientFun(preSum1)),
            (derivativeErrorValue2 * parceptron.gradientFun(preSum2)),
        )

        Assert.assertArrayEquals(expectedGradient.toTypedArray(), parceptron._gradientVectorSOOffsets.toTypedArray())
    }

    @Test
    fun `learning EXPECT check getWeightAndOffsetsGradients and second to first weights layer gradient calculation from previous gradient`() {
        repeat(100){
            val expectedOutputIndex = 1

            val parceptron = Parseptron(
                inputSize = 1,
                firstLayerSize = 3,
                secondLayerSize = 2,
                outputSize = 2
            )

            parceptron.learning(1, expectedOutputIndex)

            val gradientSum1 = parceptron._gradientVectorSOWeights[0] / parceptron.secondLayer[0] * parceptron.secondLayerWeightsValues[0][0] + parceptron._gradientVectorSOWeights[2] / parceptron.secondLayer[0] * parceptron.secondLayerWeightsValues[1][0]
            val gradientSum2 = parceptron._gradientVectorSOWeights[1] / parceptron.secondLayer[1] * parceptron.secondLayerWeightsValues[0][1] + parceptron._gradientVectorSOWeights[3] / parceptron.secondLayer[1] * parceptron.secondLayerWeightsValues[1][1]


            val preSum1 = parceptron.firstLayer[0] * parceptron.firstLayerWeightsValues[0][0] + parceptron.firstLayer[1] * parceptron.firstLayerWeightsValues[0][1] + parceptron.firstLayer[2] * parceptron.firstLayerWeightsValues[0][2] + parceptron.firstLayerOffsetsValues[0]
            val preSum2 = parceptron.firstLayer[0] * parceptron.firstLayerWeightsValues[1][0] + parceptron.firstLayer[1] * parceptron.firstLayerWeightsValues[1][1] + parceptron.firstLayer[2] * parceptron.firstLayerWeightsValues[1][2] + parceptron.firstLayerOffsetsValues[1]

            val expectedGradient = mutableListOf(
                (gradientSum1 * parceptron.gradientFun(preSum1) * parceptron.firstLayer[0]), // w1
                (gradientSum1 * parceptron.gradientFun(preSum1) * parceptron.firstLayer[1]), // w3
                (gradientSum1 * parceptron.gradientFun(preSum1) * parceptron.firstLayer[2]), // w5

                (gradientSum2 * parceptron.gradientFun(preSum2) * parceptron.firstLayer[0]), // w2
                (gradientSum2 * parceptron.gradientFun(preSum2) * parceptron.firstLayer[1]), // w4
                (gradientSum2 * parceptron.gradientFun(preSum2) * parceptron.firstLayer[2]), // w6
            )

//            println("test gradientSum1 $gradientSum1 = ${parceptron._gradientVectorSOWeights[0]} / ${parceptron.secondLayer[0]} * ${parceptron.secondLayerWeightsValues[0][0]} + ${parceptron._gradientVectorSOWeights[2]} / ${parceptron.secondLayer[0]} * ${parceptron.secondLayerWeightsValues[1][0]}")
//            println(parceptron._gradientVectorFSWeights)
//            println(expectedGradient)

            Assert.assertArrayEquals(
                expectedGradient.map { BigDecimal(it).setScale(10,RoundingMode.DOWN) }.toTypedArray(),
                parceptron._gradientVectorFSWeights.map { BigDecimal(it).setScale(10,RoundingMode.DOWN) }.toTypedArray()
            )
        }
    }
}