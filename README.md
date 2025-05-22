# Face classifiers for Android

The project contains 3 classifiers:

1. Glasses/sunglasses – detect glasses and sunglasses on a face
2. Face covering – detect face being covered, e.g., by a mask
3. ID card face – detect whether a face image on an ID card is fake or genuine

## Installation

### Repository setup

The libraries are hosted in Applied Recognition's GitHub Packages repository. Please contact Applied Recognition to obtain credentials to access the packages.

You'll obtain a user name and a token that's valid for one hour. You will also receive a Python script that will refresh the token as needed.

Store the user name and token in your local.properties file as `gpr.user` and `gpr.token` respectively. Then in your build file add the following repository in your `repositories` block:

```kotlin
maven {
    url = uri("https://maven.pkg.github.com/AppliedRecognition/Ver-ID-3D-Android-Libraries")
    credentials {
        username = settings.extra["gpr.user"] as String?
        password = settings.extra["gpr.token"] as String?
    }
}
```

### Dependencies to import

#### Glasses/sunglasses

`com.appliedrec.verid3:face-classification-glasses:1.0.0`

#### Face covering

`com.appliedrec.verid3:face-classification-covering:1.0.0`

#### ID card face

`com.appliedrec.verid3:face-classification-idcard:1.0.0`

## Usage

First, you will need to detect a face in an image. For best results import the Ver-ID face detection library from `com.appliedrec.verid3:face-landmark-detection-mp:1.0.0`. You will also need to create an image type that can be used by Ver-ID. If your source image is an Android `Bitmap` you will want to import `com.appliedrec.verid3:common-serialization:1.0.0`.

### Detect a face in image

Let's assume that you have an Android `Bitmap` in which you want to detect a face and run face classification.

```kotlin
import com.appliedrec.verid3.common.Face
import com.appliedrec.verid3.common.Image
import com.appliedrec.verid3.common.serialization.fromBitmap
import com.appliedrec.verid3.facedetection.mplandmarks.FaceDetection

suspend fun detectFaceInBitmap(context: Context, bitmap: Bitmap): Pair<Image,Face> {
    val faceDetection = FaceDetection(context)
    val image = Image.fromBitmap(bitmap)
    val face = faceDetection.detectFacesInImage(image, 1).first()
    return image to face
}
```

### Run face classification

```kotlin
// Glasses detection
// The GlassesClassificationResult class has 2 properties:
// glasses: Float and sunglasses: Float. Each property can have a value between 0 and 1. If clear glasses 
// are detected the glasses property will be > 0.5. If sunglasses are detected both the glasses and sunglasses
// properties will have a value > 0.5.
suspend fun detectGlassesInFace(context: Context, face: Face, image: Image): GlassesClassificationResult {
    // Create an instance of the GlassesClassifier
    // The create factory method is suspending, you need to call it in a coroutine context
    val classifier = GlassesClassifier.create(context)
    // The classifier classes implement the AutoCloseable interface, either call the extractAttribute
    // method in a use block or call close() on the classifier once you're done with it. This ensures 
    // the classifier resources are cleaned up after use.
    return classifier.use {
        it.extractAttribute(face, image)
    }
}

// Face covering detection
// The face covering classifier returns a value between 0 and 1. 1 means highest confidence that
// the face is covered.
suspend fun detectFaceCoveringInFace(context: Context, face: Face, image: Image): Float {
    // Create an instance of the FaceCoveringClassifier
    // The create factory method is suspending, you need to call it in a coroutine context
    val classifier = FaceCoveringClassifier.create(context)
    // The classifier classes implement the AutoCloseable interface, either call the extractAttribute
    // method in a use block or call close() on the classifier once you're done with it. This ensures 
    // the classifier resources are cleaned up after use.
    return classifier.use {
        it.extractAttribute(face, image)
    }
}

// Fake ID image detection
// The classifier returns a value between 0 and 1. 1 means highest confidence that the face is 
// genuine part of the ID card. 0 means it's most likely tampered.
suspend fun detectGenuineIdCardFace(context: Context, face: Face, image: Image): Float {
    // Create an instance of the IdCardFaceClassifier
    // The create factory method is suspending, you need to call it in a coroutine context
    val classifier = IdCardFaceClassifier.create(context)
    // The classifier classes implement the AutoCloseable interface, either call the extractAttribute
    // method in a use block or call close() on the classifier once you're done with it. This ensures 
    // the classifier resources are cleaned up after use.
    return classifier.use {
        it.extractAttribute(face, image)
    }
}
```