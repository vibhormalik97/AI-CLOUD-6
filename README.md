# AIOPS-Assignment-6
Created to as part of CU IEOR 4577 - AI in the Cloud.

This assignment marks the completion of our 6 week project where in we built a Big Data Pipeline/App on AWS.
The app uses a CNN model trained on 10M tweet to identify the sentiment of tweets. It can be used by the following REST API call.

```ruby
curl -X POST https://6f47m3p3bl.execute-api.us-east-1.amazonaws.com/v1/predict --header "Content-Type:application/json" --data '{"tweet": "Sample tweet"}'
```
