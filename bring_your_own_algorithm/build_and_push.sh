%%sh

algorithm_name=sagemaker-lightgbm
 
chmod +x lightgbm/train
chmod +x lightgbm/serve
 
account=$(aws sts get-caller-identity --query Account --output text)
 
region=$(aws configure get region)
region=${region:-us-west-2}


fullname="${account}.dkr.ecr.${region}.amazonaws.com/${algorithm_name}:latest"
 
aws --region ${region} ecr describe-repositories --repository-names "${algorithm_name}" > /dev/null 2>&1
 
if [ $? -ne 0 ]
then
    aws --region ${region} ecr create-repository --repository-name "${algorithm_name}" > /dev/null
fi
 
$(aws ecr get-login --region ${region} --no-include-email)
 
docker build  -t ${algorithm_name} .
docker tag ${algorithm_name} ${fullname}
 
docker push ${fullname}