package com.appliedrec.verid3.faceclassification

import android.content.Context
import com.appliedrec.verid3.common.Face
import com.appliedrec.verid3.common.IImage
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.withContext
import java.util.concurrent.atomic.AtomicBoolean

abstract class FaceClassifier<T> protected constructor(private val nativeContext: Long) : AutoCloseable {

    private var isClosed = AtomicBoolean(false)

    companion object {
        init {
            System.loadLibrary("FaceClassification")
        }

        @JvmStatic
        suspend fun createContext(context: Context, name: String, modelName: String): Long {
            val modelBytes = readModelFile(context, modelName)
            return createNativeContext(name, modelBytes)
        }

        @JvmStatic
        private suspend fun readModelFile(context: Context, path: String): ByteArray =
            withContext(Dispatchers.IO) {
                context.assets.open(path).readBytes()
            }

        @JvmStatic
        private external fun createNativeContext(name: String, modelBytes: ByteArray): Long
    }

    suspend fun extractAttribute(face: Face, image: IImage): T = coroutineScope {
        require(!isClosed.get())
        val result = extractAttribute(nativeContext, face, image)
        transformResult(result)
    }

    private external fun extractAttribute(context: Long, face: Face, image: IImage): FloatArray

    private external fun destroyNativeContext(context: Long)

    protected abstract fun transformResult(result: FloatArray): T

    override fun close() {
        if (isClosed.compareAndSet(false, true)) {
            destroyNativeContext(nativeContext)
        }
    }
}