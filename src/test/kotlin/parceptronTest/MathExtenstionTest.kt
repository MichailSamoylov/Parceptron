package parceptronTest

import org.junit.Assert
import org.junit.Test
import parseptron.elementWiseMultiplication
import parseptron.matrixMultiplication
import parseptron.transposition
import parseptron.vectorWiseMultiplication

class MathExtenstionTest {

    @Test
    fun `transpositing matrix EXPECT matrix is transposited`() {
//        val matrix = mutableListOf(
//            mutableListOf(1.0, 2.0, 3.0),
//            mutableListOf(4.0, 5.0, 6.0),
//        )
//
//        val expected = mutableListOf(
//            mutableListOf(1.0, 4.0),
//            mutableListOf(2.0, 5.0),
//            mutableListOf(3.0, 6.0),
//        )

        val matrix = mutableListOf(
            mutableListOf(1.0, 2.0),
            mutableListOf(3.0, 4.0),
        )

        val expected = mutableListOf(
            mutableListOf(1.0, 3.0),
            mutableListOf(2.0, 4.0),
        )

        val actual = matrix.transposition()

        Assert.assertEquals(expected, actual)
    }

    @Test
    fun `matrix multiplication EXPECT vector`() {
        val matrixLeft = mutableListOf(
            mutableListOf(2.0, -3.0, 1.0),
            mutableListOf(5.0, 4.0, -2.0),
        )

        val matrixRight = mutableListOf(-7.0, 2.0, 4.0)

        val expectedVector = mutableListOf(-16.0, -35.0)

        val actual = matrixMultiplication(matrixLeft, matrixRight)

        Assert.assertEquals(expectedVector, actual)
    }

    @Test
    fun `element wise multiplication EXPECT each element of matrix multiplied on value`() {
        val matrix = mutableListOf(
            mutableListOf(2.0, -3.0, 1.0),
            mutableListOf(5.0, 4.0, -2.0),
        )
        val value = 2.0

        val expectedMatrix = mutableListOf(
            mutableListOf(4.0, -6.0, 2.0),
            mutableListOf(10.0, 8.0, -4.0),
        )

        val actual = matrix.elementWiseMultiplication(value)

        Assert.assertEquals(expectedMatrix, actual)
    }

    @Test
    fun `vector wise multiplication EXPECT vectors multiplied`() {
        val vector1 = mutableListOf(2.0, -3.0, 1.0)
        val vector2 = mutableListOf(5.0, 4.0, -2.0)

        val expectedMatrix = mutableListOf(10.0, -12.0, -2.0)

        val actual = vector1.vectorWiseMultiplication(vector2)

        Assert.assertEquals(expectedMatrix, actual)
    }
}