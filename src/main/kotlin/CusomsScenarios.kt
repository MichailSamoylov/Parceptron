import parseptron.Parseptron
import java.io.File
import javax.imageio.ImageIO

fun learnAndTest(parseptron: Parseptron) {
//	val symbol = 6
//	for (p in 1..1000) {
//
//		val image = ImageIO.read(File("D:\\IntelliJ\\parseptron\\dataset\\MNIST - JPG - training\\$symbol\\p ($p).jpg"))
//		parseptron.learning(image, symbol)
//
//		println("learning symbol $symbol example $p from 2")
//	}
//
//	val image = ImageIO.read(File("D:\\IntelliJ\\parseptron\\dataset\\MNIST - JPG - testing\\$symbol\\p (50).jpg"))
//	println(parseptron.recognize(image))
//
//	val image = ImageIO.read(File("D:\\IntelliJ\\parseptron\\dataset\\MNIST - JPG - testing\\3\\p (1).jpg"))
//	println(parseptron.recognize(image))
//
//	val examplesCount = 2000
//
//	for (p in 1..examplesCount) {
//
//		for (i in 0..9) {
//
//			val image = ImageIO.read(File("D:\\IntelliJ\\parseptron\\dataset\\MNIST - JPG - training\\$i\\p ($p).jpg"))
//			parseptron.learning(image, i)
//
//			//println("learning symbol $i example $p from $examplesCount")
//		}
//	}
//
//	var testingErrorCount = 0
//	for (p in 1..890) {
//
//		for (i in 0..9) {
//
//			val image = ImageIO.read(File("D:\\IntelliJ\\parseptron\\dataset\\MNIST - JPG - testing\\$i\\p ($p).jpg"))
//
//			println("testing symbol $i example $p from 890")
//			val result = parseptron.recognize(image)
//			println(result)
//
//			if (result != i) {
//
//				testingErrorCount++
//			}
//		}
//	}
//
//	println(testingErrorCount)
//
//	println(parseptron.outputLayer.toList())
}

fun test(parseptron: Parseptron) {
    val symbol = 6
    val image = ImageIO.read(File("D:\\IntelliJ\\parseptron\\dataset\\MNIST - JPG - testing\\$symbol\\p (50).jpg"))
    println(parseptron.recognize(image))
}