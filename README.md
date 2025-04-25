# CS4296_Project
Method 1:
1.	Create a AWS S3 bucket and EC2 instance
2.	Copy the python code to the EC2 instance
3.	Install all the packages and set the permission of EC2 access the AWS bucket
4.	Syncing the S3 bucket to EC2 instance
5.	Run the python code and get the result
-docker file only for the code (not inculde images and EC2 instance), if can't run please copy and copy and paste

Method 2:
1.	Create a new role on AWS IAM with full access to rekognition
2.	Get the Access Key ID and Secret access key
3.	Install neccessary packages and use the API connection to grant the service
4.	Run the python code and get the result

Method 3:
1.	Download an original pytorch model file on Hugging Face
2.	Write a python script to do the image recognition task
3.	Run the python code on local and get the result
