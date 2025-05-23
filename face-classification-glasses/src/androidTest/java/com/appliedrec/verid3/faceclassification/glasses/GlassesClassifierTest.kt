package com.appliedrec.verid3.faceclassification.glasses

import android.graphics.BitmapFactory
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.appliedrec.verid3.common.Image
import com.appliedrec.verid3.common.serialization.fromBitmap
import com.appliedrec.verid3.facedetection.mplandmarks.FaceDetection
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Instrumented test, which will execute on an Android device.
 *
 * See [testing documentation](http://d.android.com/tools/testing).
 */
@RunWith(AndroidJUnit4::class)
class GlassesClassifierTest {

    @Test
    fun testCreateClassifier() = runBlocking {
        val appContext = InstrumentationRegistry.getInstrumentation().targetContext
        val classifier = GlassesClassifier.create(appContext)
        assertNotNull(classifier)
    }

    @Test
    fun testDetectGlasses() = runBlocking {
        val result = runDetection("glasses.jpg")
        assertTrue(result.glasses > 0.5)
        assertTrue(result.sunglasses < 0.5)
    }

    @Test
    fun testDetectSunglasses() = runBlocking {
        val result = runDetection("sunglasses.jpg")
        assertTrue(result.glasses > 0.5)
        assertTrue(result.sunglasses > 0.5)
    }

    @Test
    fun testDetectNoGlasses() = runBlocking {
        val result = runDetection("noglasses.jpg")
        assertTrue(result.glasses < 0.5)
        assertTrue(result.sunglasses < 0.5)
    }

    private suspend fun runDetection(imageName: String): GlassesClassificationResult {
        val appContext = InstrumentationRegistry.getInstrumentation().targetContext
        val classifier = GlassesClassifier.create(appContext)
        val faceDetection = FaceDetection(appContext)
        val image = appContext.assets.open(imageName).use {
            val bitmap = BitmapFactory.decodeStream(it)
            Image.fromBitmap(bitmap)
        }
        val face = faceDetection.detectFacesInImage(image, 1).first()
        return classifier.use { it.extractAttribute(face, image) }
    }
}