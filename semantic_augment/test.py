from util import  *
import os

def test_cal_tag_tag_sim():
    fs = os.listdir('query_pkl')[:5]
    cal_tag_tag_sim(fs)

def test_plot_query():
    total = []
    cuts = []
    for i, fn in enumerate(os.listdir('query_pkl')):
        #print(i)
        tmp = get_query_result('query_pkl/' + fn)
        total.extend(tmp)
        gradient = np.gradient(sorted(tmp))
        cutoff = len(np.where(gradient > 0.005)[0])
        cuts.append(cutoff)
        try :
            check_wierd_query('query_pkl/' + fn)
        except:
            continue

    indices = np.argsort(cuts)

    #print(np.where(gradient > 0.005))
    plt.plot(cuts)
    #plt.plot(sorted(total))
    #plt.axvline(1000-cutoff)
    #print(np.gradient(sorted(total)))

    #print(np.gradient(np.gradient(sorted(total))))
    plt.savefig('hist.png')
    # plt.xlabel('1000 most similar words')
    # plt.ylabel('Similarity')
    # plt.savefig('sim_dynmaics_1000_examples.jpg')
    # plt.close()
    plt.close()

def test_plot_ind():
    plt.close()
    fns = os.listdir('query_pkl')
    f1 = fns[12]
    f2 = fns[112]
    #print(f1, f2)

    d1 = pickle.load(open('query_pkl/' + f1, 'rb'))
    d2 =  pickle.load(open('query_pkl/' + f2, 'rb'))
    print(f1, d1)
    print(f2, d2)
    #r2 = plot_query_result('query_pkl/' + f2, f2.replace('.pkl', '.jpg'))
    #plt.plot(r2)
    #plt.savefig('s.png')

if __name__ == '__main__':
    #calculate_idf_all()
    #test_plot_query()
    test_cal_tag_tag_sim()

# set the threshold of hist.png to update the T-T matrix