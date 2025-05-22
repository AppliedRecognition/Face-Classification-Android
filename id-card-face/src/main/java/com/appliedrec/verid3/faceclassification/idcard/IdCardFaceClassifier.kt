package com.appliedrec.verid3.faceclassification.idcard

import android.content.Context
import com.appliedrec.verid3.faceclassification.FaceClassifier

class IdCardFaceClassifier private constructor(nativeContext: Long) :
    FaceClassifier<Float>(nativeContext) {

    override fun transformResult(result: FloatArray): Float {
        require(result.isNotEmpty())
        return result[0]
    }

    companion object {
        @JvmStatic
        suspend fun create(context: Context): IdCardFaceClassifier {
            val modelFileName = "license01-20210820ay-xiypjic%2200-q08.nv"
            val nativeContext = FaceClassifier.createContext(
                context,
                "IdCardFace",
                modelFileName
            )
            return IdCardFaceClassifier(nativeContext)
        }
    }
}