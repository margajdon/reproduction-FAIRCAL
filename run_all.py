from cosine_similarity_calcs import derive_cosine_sim_for_all_sets
from csv_creator import create_rfw_csv_template, create_similarity_data
from generate_embeddings import generate_all_embeddings
from main_fairness_analysis import run_complete_analysis


def run_entire_pipeline():
    """
    This function runs the entire pipeline consisting in the following steps:
    1. Creation of the template for the RFW dataset
    2. Embedding generation for the RFW and BFW dataset and their respective models
    3. Derivation of the cosine similarities between the embedding pairs
    4. Reformatting of the similarity data to be used for the fairness analysis
    5. Run the fairness analysis under all settings
    """
    # Create a template for the RFW dataset
    create_rfw_csv_template()
    # Generate the embeddings for the RFW and BFW dataset and their respective models
    generate_all_embeddings()
    # Derive the cosine similarities
    derive_cosine_sim_for_all_sets()
    # Reformat the similarity data to be used for the fairness analysis
    create_similarity_data()
    # Run the fairness analysis under all settings
    run_complete_analysis()


if __name__ == '__main__':
    # Run the entire pipeline
    run_entire_pipeline()


