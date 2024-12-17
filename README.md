# Potential Talents
This project implements a Learning to Rank (LTR) solution for recruiting candidates based on their relevance to specific job search terms. The model leverages neural networks trained on GloVe embeddings to rank candidates by their suitability for the recruiter’s criteria.

## Objective

To build a scalable and effective solution for recruiters to rank candidates based on job title embeddings and search term similarities.

## Key Pieces
Learning to Rank (LTR): A neural network architecture trained to rank candidates effectively.
Embedding Models: Utilizes GloVe embeddings to represent candidate job titles and search terms in a vector space.
Performance Metrics: Evaluates results using Kendall's Tau, Spearman's Rank Correlation, and Normalized Discounted Cumulative Gain (NDCG).
Cross-validation: Implements 5-fold cross-validation for robust model evaluation.

## Data Description

The dataset contains candidate information (features) from LinkedIn like job title, location and number of connections. However, job title is the only useful feature for the project objective.

Number of examples: 104

## Data Preprocessing
Duplicates: All duplicate job titles need to be removed to ensure data quality.
Useful Feature: Only the job_title column is utilized for generating embeddings and relevance rankings, as it aligns with the problem's objective.

## Embedding Source
Pre-trained GloVe embeddings are used to convert job_title strings into vector representations for training the neural network.

## Environment Requirements
This project is designed to run in Google Colab, leveraging its GPU capabilities for training the neural network. Ensure that Project3.ipynb and potential-talents.csv are in the same directory. Ensure that root variable in the notebook is correctly set to the directory path. Finally, download the glove embeddings (glove.6b.50d.txt) to the directory. You can find them here: https://nlp.stanford.edu/data/glove.6B.zip.

## Observations
Performance: High-performance metrics across folds (NDCG > 98%).
Scalability: The current solution requires a separate model for each search term, limiting scalability for multiple terms.
Improvement Potential: Fine-tuning an LLM and incorporating more candidate data (e.g., detailed job descriptions) can further enhance the model.

## Conclusion
The deliverable is an LTR neural network tailored for specific search terms, which serves as a foundation for a scalable ranking solution. Specifically, the LTR neural network should be trained on 50 dimensional glove embeddings for the job titles and the cosine similarities to each search term. This implies that there is a separate LTR neural network for each search term. While the current approach works effectively for limited terms, future extensions can address scalability and data richness to improve utility for recruiters.