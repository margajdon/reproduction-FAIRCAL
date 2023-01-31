from compare_cosine import derive_cosine_sim_for_all_sets
from csv_creator import create_rfw_csv_template, create_similarity_data
from generate_embeddings import generate_all_embeddings
from main import run_complete_analysis

if __name__ == '__main__':
    create_rfw_csv_template()
    generate_all_embeddings()
    derive_cosine_sim_for_all_sets()
    create_similarity_data()
    run_complete_analysis()

