pipeline {
    agent any

    stages {
        stage('Checkout') {
            steps {
                // Github 소스코드 가져오기
                checkout scm
            }
        }

        stage('Build') {
            steps {
                echo 'building the application...'
                // 예: mvn clean package 또는 npm install
            }
        }

        stage('Test') {
            steps {
                echo 'testing the application...'
                // 예: mvn test 또는 pytest
            }
        }

        stage('Docker Build') {
            steps {
                // Docker 이미지 빌드 (이미지 이름: langchain-practice, 태그: latest)
                sh 'docker build -t amdp-registry.skala-ai.com/skala26a-ai2/langchain-practice:latest .'
            }
        }

        stage('Push to Harbor') {
            steps {
                // Jenkins Credential에 등록된 harbor-cred 사용
                withCredentials([usernamePassword(credentialsId: 'harbor-cred', usernameVariable: 'HARBOR_USER', passwordVariable: 'HARBOR_PASS')]) {
                    sh 'docker login amdp-registry.skala-ai.com -u $HARBOR_USER -p $HARBOR_PASS'
                    sh 'docker push amdp-registry.skala-ai.com/skala26a-ai2/langchain-practice:latest'
                }
            }
        }

    }
}
