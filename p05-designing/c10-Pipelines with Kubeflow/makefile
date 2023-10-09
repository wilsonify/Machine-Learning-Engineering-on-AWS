
clean:
	kubectl port-forward svc/istio-ingressgateway -n istio-system --address=localhost 8080:80
	cd ~/environment/ch10/kubeflow-manifests/
	cd deployments/vanilla/
	kustomize build . | kubectl delete -f -
	eksctl delete cluster --region $CLUSTER_REGION --name $CLUSTER_NAME

Cloud9:
	aws cloud9 update-environment --managed-credentials-action DISABLE --environment-id $C9_PID
	rm -vf /home/ubuntu/.aws/credentials
	aws sts get-caller-identity --query Arn

prerequisites:
	wget -O prerequisites.zip https://bit.ly/3ByyDGV
	unzip prerequisites.zip
	cd ch10_prerequisites
	chmod +x *.sh
	sudo ./00_install_kubectl_aws_jq_and_more.sh
	aws --version
	kustomize version
	eksctl version
	~/.bash_completion
	~/.bash_profile
	~/.bashrc
	export AWS_REGION="us-west-2"
	echo "export AWS_REGION=${AWS_REGION}" | tee -a ~/.bash_profile
	aws configure set default.region ${AWS_REGION}
	aws configure get default.region

Kubeflow:
	cd ~/environment
	mkdir ch10
	cd ch10
	touch eks.yaml

eks:
	export CLUSTER_NAME=kubeflow-eks-000
	export CLUSTER_REGION=us-west-2
	export KUBEFLOW_VERSION=v1.5.1
	export AWS_VERSION=v1.5.1-aws-b1.0.0
	eksctl create cluster -f eks.yaml --dry-run
	eksctl create cluster -f eks.yaml
	kubectl get nodes -o wide
	eksctl utils associate-iam-oidc-provider --cluster $CLUSTER_NAME --approve -v4
	aws eks update-kubeconfig --name $CLUSTER_NAME --region ${AWS_REGION}
	git clone https://github.com/awslabs/kubeflow-manifests.git && cd kubeflow-manifests
	git checkout ${AWS_VERSION}
	git clone --branch ${KUBEFLOW_VERSION} https://github.com/kubeflow/manifests.git upstream
	cd deployments/vanilla
	while ! kustomize build . | kubectl apply -f -; do echo "Retrying"; sleep 30; done
	ns_array=(kubeflow kubeflow-user-example-com kserve cert-manager istio-system auth knative-eventing knative-serving)
	for i in ${ns_array[@]}; do
		echo "[+] kubectl get pods -n $i"
		kubectl get pods -n $i;
		echo "---"
	done
	kubectl port-forward svc/istio-ingressgateway -n istio-system 8080:80 --address=localhost
