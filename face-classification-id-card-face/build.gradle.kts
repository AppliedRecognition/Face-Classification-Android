plugins {
    alias(libs.plugins.android.library)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.dokka)
    `maven-publish`
    signing
}

android {
    namespace = "com.appliedrec.verid3.faceclassification.idcard"
    compileSdk = 35

    defaultConfig {
        minSdk = 26

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        consumerProguardFiles("consumer-rules.pro")
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    kotlinOptions {
        jvmTarget = "11"
    }
}

dependencies {

    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.appcompat)
    api(project(":face-classification-core"))
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
    androidTestImplementation(libs.verid.facedetection.mp.landmarks)
    androidTestImplementation(libs.verid.common.serialization)
}

tasks.withType<org.jetbrains.dokka.gradle.DokkaTask>().configureEach {
    moduleName.set("ID card face")
}

publishing {
    publications {
        create<MavenPublication>("release") {
            groupId = "com.appliedrec.verid3"
            artifactId = "face-classification-id-card-face"
            version = "1.0.0"
            afterEvaluate {
                from(components["release"])
            }
            pom {
                name.set("Genuine ID card face image detection")
                description.set("Detects whether a face image on an ID card is fake or genuine")
                url.set("https://github.com/AppliedRecognition/Face-Classification-Android")
                licenses {
                    license {
                        name.set("Commercial")
                        url.set("https://raw.githubusercontent.com/AppliedRecognition/Face-Classification-Android/main/LICENCE.txt")
                    }
                }
                developers {
                    developer {
                        id.set("appliedrec")
                        name.set("Applied Recognition")
                        email.set("support@appliedrecognition.com")
                    }
                }
                scm {
                    connection.set("scm:git:git://github.com/AppliedRecognition/Face-Classification-Android.git")
                    developerConnection.set("scm:git:ssh://github.com/AppliedRecognition/Face-Classification-Android.git")
                    url.set("https://github.com/AppliedRecognition/Face-Classification-Android")
                }
            }
        }
    }

    repositories {
        maven {
            name = "GitHubPackages"
            url = uri("https://maven.pkg.github.com/AppliedRecognition/Ver-ID-Releases-Android")
            credentials {
                username = project.findProperty("gpr.user") as String?
                password = project.findProperty("gpr.token") as String?
            }
        }
    }
}

signing {
    useGpgCmd()
    sign(publishing.publications["release"])
}