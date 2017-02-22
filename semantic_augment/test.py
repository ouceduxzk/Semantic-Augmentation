from util import  *
import os


def test_plot_query():
    for i, fn in enumerate(os.listdir('query_pkl')):
        print(i)
        plot_query_result('query_pkl/' + fn, fn.replace('.pkl', '.jpg'))
        try :
            check_wierd_query('query_pkl/' + fn)
        except:
            continue

    plt.xlabel('1000 most similar words')
    plt.ylabel('Similarity')
    plt.savefig('sim_dynmaics_1000_examples.jpg')
    plt.close()

if __name__ == '__main__':
    #calculate_idf_all()
    test_plot_query()