import json
import boto3
import email
import os

# import the utilities functions( from the tutorial)
from sms_spam_classifier_utilities import one_hot_encode, vectorize_sequences

# get the s3 instance and sageMaker
s3 = boto3.client('s3')
runtime = boto3.client('runtime.sagemaker')
email_client = boto3.client('ses')

# function to handel the s3 event trigger
def lambda_handler(event, context):
    
    # get the email from bucket and key
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']
    response = s3.get_object(Bucket=bucket, Key=key)
    
    # read the content of the email
    email_item = response['Body'].read()
    email_contents = email.message_from_bytes(email_item)
    body = email_contents.get_payload()[0].get_payload()
    
    # get email details
    received_date = email_contents["Date"]
    email_subject = email_contents["Subject"]
    from_address = email_contents["From"]
    to_address = email_contents["To"]
    # from_address = email_item.get('From')

    # set the notebook endpoint
    endpoint = 'sms-spam-classifier-mxnet-2021-11-18-06-51-19-466'
    endpoint_name = os.environ.get('e1', endpoint)

    # preprocess the email data
    body = body.strip()
    vocabulary_length = 9013
    text = [body]
    one_hot_messages = one_hot_encode(text, vocabulary_length)
    encoded_messages = vectorize_sequences(one_hot_messages, vocabulary_length)
    payload = json.dumps(encoded_messages.tolist())
    
    # call sagemaker endpoint
    response = runtime.invoke_endpoint(EndpointName=endpoint_name, ContentType='application/json', Body=payload)
    result = json.loads(response["Body"].read())
    prediction = result['predicted_label'][0][0]
    
    # check if email was spam or okay using model and calculate confidence score
    if prediction == 0:
        label = 'Ok'
    else:
        label = 'Spam'
    confidence_score = float(result['predicted_probability'][0][0]) * 100

    # create a message to send back
    message = "We received your email sent at " + str(received_date) + " with the subject " + str(email_subject) + ".\nHere \is a 240 character sample of the email body:\n\n" + body[:240] + "\nThe email was \ categorized as " + str(label) + " with a " + str(confidence_score) + "% confidence."
    
    # send a response email
    response_email = email_client.send_email(
        Destination={'ToAddresses': [from_address]},
        Message={
            'Body': {
                'Text': {
                    'Charset': 'UTF-8',
                    'Data': message,
                },
            },
            'Subject': {
                'Charset': 'UTF-8',
                'Data': 'Spam analysis of your email',
            },
        },
        Source=str(email_contents.get('To')),
    )
    print(response_email)
    
    # return body(for loggin)
    return {
        'statusCode': 200,
        'body': json.dumps(body)
    }
    
