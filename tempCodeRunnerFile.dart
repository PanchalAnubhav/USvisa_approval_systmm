MongoDB Collections:
1.	visa_applications Collection:

{
  "_id": ObjectId,
  "case_id": String,
  "employer_name": String,
  "soc_name": String,
  "job_title": String,
  "full_time_position": Boolean,
  "prevailing_wage": Number,
  "year": Number,
  "worksite_postal_code": String,
  "case_status": String,
  "created_at": DateTime,
  "updated_at": DateTime
}
2. predictions Collection:
{
  "_id": ObjectId,
  "prediction_id": String,
  "input_features": Object,
  "prediction": String,
  "confidence_score": Number,
  "model_version": String,
  "timestamp": DateTime
}
3. models Collection:
{
  "_id": ObjectId,
  "model_id": String,
  "model_name": String,
  "version": String,
  "accuracy": Number,
  "training_date": DateTime,
  "status": String,
  "file_path": String
}
