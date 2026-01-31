pipeline {
    agent any

    environment {
        VENV_NAME = "venv"
        MLFLOW_TRACKING_URI = "http://localhost:5555"
        MLFLOW_EXPERIMENT_NAME = "Real-Estate-Price"
        API_PORT = "7000"
    }

    stages {

        /* ================================
           Stage 1: Checkout
        ================================= */
        stage("Checkout Code") {
            steps {
                git branch: "master",
                    url: "https://github.com/anjilinux/project-mlflow-Real-Estate-Price-Prediction-Model-with-NLP-FastAPI.git"
            }
        }

        /* ================================
           Stage 2: Virtual Environment
        ================================= */
        stage("Setup Virtual Environment") {
            steps {
                sh '''
                python3 -m venv $VENV_NAME
                . $VENV_NAME/bin/activate

                pip install -r requirements.txt
                '''
            }
        }

        /* ================================
           Stage 3: Data Ingestion
        ================================= */
        stage("Data Ingestion") {
            steps {
                sh '''
                . $VENV_NAME/bin/activate
                python data_ingestion.py
                '''
            }
        }

        /* ================================
           Stage 4: EDA & Feature Engineering
        ================================= */
        stage("EDA & Feature Engineering") {
            steps {
                sh '''
                . $VENV_NAME/bin/activate
                python eda_feature_engineering.py
                '''
            }
        }

        /* ================================
           Stage 5: Data Preprocessing
        ================================= */
        stage("Data Preprocessing") {
            steps {
                sh '''
                . $VENV_NAME/bin/activate
                python preprocess.py
                '''
            }
        }

        /* ================================
           Stage 6: Model Training
        ================================= */
        stage("Model Training") {
            steps {
                sh '''
                . $VENV_NAME/bin/activate
                python train.py
                '''
            }
        }

        /* ================================
           Stage 7: Model Evaluation
        ================================= */
        stage("Model Evaluation") {
            steps {
                sh '''
                . $VENV_NAME/bin/activate
                python evaluate.py
                '''
            }
        }

        /* ================================
           Stage 8: Pytest
        ================================= */
        stage("Model Testing") {
            steps {
                sh '''
                . $VENV_NAME/bin/activate
                pytest test_model.py
                '''
            }
        }

        /* ================================
           Stage 9: Prediction Smoke Test
        ================================= */
        stage("Prediction Test") {
            steps {
                sh '''
                . $VENV_NAME/bin/activate
                python predict.py
                '''
            }
        }


stage("FastAPI API Test") {
    steps {
        sh '''
        set -e
        . $VENV_NAME/bin/activate

        nohup uvicorn src.api.main:app --host 0.0.0.0 --port $API_PORT > api.log 2>&1 &
        API_PID=$!
        sleep 10

        curl -sf http://localhost:$API_PORT/health

        RESPONSE=$(curl -s -X POST http://localhost:$API_PORT/predict \
          -H "Content-Type: application/json" \
          -d '{
                "area": 1200,
                "bhk": 2,
                "bath": 2,
                "description": "luxury apartment near metro"
              }')

        echo "API Response: $RESPONSE"
        kill -9 $API_PID
        '''
    }
}





















        // stage("FastAPI API Test") {
        //     steps {
        //         sh '''
        //         set -e
        //         . $VENV_NAME/bin/activate

        //         nohup uvicorn src.api.main:app --host 0.0.0.0 --port $API_PORT > api.log 2>&1 &
        //         API_PID=$!
        //         sleep 10

        //         curl -sf http://localhost:$API_PORT/health

        //         RESPONSE=$(curl -s -X POST http://localhost:$API_PORT/predict \
        //           -H "Content-Type: application/json" \
        //           -d '{
        //                 "area": 1200,
        //                 "bhk": 2,
        //                 "bath": 2,
        //                 "description": "luxury apartment near metro"
        //               }')

        //         echo "API Response: $RESPONSE"
        //         kill -9 $API_PID
        //         '''
        //     }
        // }

        /* ================================
           Stage 11: Docker Build & Test
        ================================= */
        stage("Docker Build & Run") {
            steps {
                sh '''
                docker build -t real-estate-api .
                docker run -d -p 8006:8005 --name real-estate-api real-estate-api
                sleep 15
                curl -sf http://localhost:8001/health
                docker stop real-estate-api
                docker rm real-estate-api
                '''
            }
        }

        /* ================================
           Stage 12: Archive Artifacts
        ================================= */
        stage("Archive Artifacts") {
            steps {
                archiveArtifacts artifacts: '*.pkl', fingerprint: true
            }
        }
    }

    post {
        success {
            echo "✅ Real Estate Price Prediction MLOps Pipeline Completed Successfully"
        }
        failure {
            echo "❌ Pipeline Failed – Check Logs"
        }
    }
}
