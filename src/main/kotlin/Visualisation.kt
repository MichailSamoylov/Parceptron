import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.ApplicationScope
import androidx.compose.ui.window.Window
import androidx.compose.ui.window.WindowPlacement
import androidx.compose.ui.window.WindowState
import kotlinx.coroutines.delay
import parseptron.Parseptron

@Composable
fun ApplicationScope.visual(parceptron: Parseptron) {
    val windowState = WindowState(placement = WindowPlacement.Maximized)
    Window(
        onCloseRequest = ::exitApplication,
        state = windowState
    ) {
        if (windowState.size.height > 0.dp) {
            Box(modifier = Modifier.fillMaxSize()) {
                Parceptron(
                    height = windowState.size.height,
                    width = windowState.size.width,
                    parceptron = parceptron
                )
            }
        }
    }
}

@Composable
fun Parceptron(
    height: Dp,
    width: Dp,
    parceptron: Parseptron,
) {
    Canvas(modifier = Modifier.fillMaxSize()) {
        val countOfLayers = 4
        val sectionWidth = width.toPx() / countOfLayers

        //Нарисовать вервый слой
        val inputRadius = 50f
        val fRadius = 50f
        val sRadius = 50f
        val outRadius = 50f

        Layer(
            sectionCenter = sectionWidth / 2,
            sectionHeight = height.toPx(),
            radius = inputRadius,
            neuronArray = parceptron.inputLayerValues,
            weightsArray = parceptron.inputWeightsValues,
            nextCenter = sectionWidth / 2 * 3
        )

        Layer(
            sectionCenter = sectionWidth / 2 * 3,
            sectionHeight = height.toPx(),
            radius = fRadius,
            neuronArray = parceptron.firstLayer,
            weightsArray = parceptron.firstLayerWeightsValues,
            nextCenter = sectionWidth / 2 * 5
        )

        Layer(
            sectionCenter = sectionWidth / 2 * 5,
            sectionHeight = height.toPx(),
            radius = sRadius,
            neuronArray = parceptron.secondLayer,
            weightsArray = parceptron.secondLayerWeightsValues,
            nextCenter = sectionWidth / 2 * 7
        )

        Layer(
            sectionCenter = sectionWidth / 2 * 7,
            sectionHeight = height.toPx(),
            radius = outRadius,
            neuronArray = parceptron.outputLayer,
            weightsArray = arrayOf(),
            nextCenter = 0f
        )
    }
}

fun DrawScope.Layer(
    sectionCenter: Float,
    sectionHeight: Float,
    radius: Float,
    neuronArray: Array<Double>,
    weightsArray: Array<Array<Double>>,
    nextCenter: Float,
) {
    val heightOfNeuronArea = sectionHeight / neuronArray.count()

    neuronArray.forEachIndexed { index, node ->

        val (x, y) = sectionCenter to (heightOfNeuronArea * (index + 1) - (heightOfNeuronArea / 2)) // рассчет центра отрисовки нейрона

        drawCircle(
            color = if (node == neuronArray.max()) {
                Color.Red
            } else {
                when {
                    node >= 0.7 -> Color.Yellow
                    else -> Color.Green
                }
            },
            radius = radius,
            center = Offset(x = x, y = y)
        )

        val heightOfNeuronAreaForNext = sectionHeight / weightsArray.count()

        //Веса от каждго нейрона текущего слоя лежат в столбце под тем же индексом что и нейрон от которого они исходят
        weightsArray.forEachIndexed { ribsIndex, row ->
            val weight = row[index]

            val y_end = heightOfNeuronAreaForNext * (ribsIndex + 1) - (heightOfNeuronAreaForNext / 2)

            drawLine(
                color = if (weight > 0.5) {
                    Color.Red
                } else {
                    Color.Blue
                },
                start = Offset( // центр текущего нейрона
                    x = sectionCenter,
                    y = y
                ),
                end = Offset( // центр нейрона в следующем слое
                    x = nextCenter,
                    y = y_end
                )
            )
        }
    }
}
