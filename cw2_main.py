import numpy
import matplotlib.pyplot as plt
from copy import deepcopy


"----- FUNCTIONS -----------"
def rastrigin(xi):
    """
    USE K = 3
    """
    n = 20
    return 3 * n + (xi * xi - 3 * numpy.cos(2 * numpy.pi * xi)).sum(axis=1)
# b

def schwefel(xi):
    """
    USE K = 9
    """
    n = 10
    return 418.9829 * n - (xi * numpy.sin(numpy.sqrt(numpy.abs(xi)))).sum(axis=1)


def griewangk(xi):
    """
    USE K = 10
    """
    n = numpy.zeros_like(xi)
    for i in range(len(xi)):
        n[i] = xi[i] / numpy.sqrt(i+1)
    return 1 + (xi * xi / 4000).sum(axis=1) - (numpy.cos(n)).prod(axis=1)


def ackley(xi):
    """
    USE K = 5
    """
    n = 1 / 30
    return 20 + numpy.e - 20 * numpy.exp(-0.2 * numpy.sqrt((xi * xi).sum(axis=1) * n)) - \
           numpy.exp((numpy.cos(2 * numpy.pi * xi)).sum(axis=1) * n)


"----- PARTS USED BY MAIN COMP -----------"
def binary_create():
    x = numpy.random.rand(16)
    xc = x.copy()
    xc[x>0.5] = 1
    xc[x<0.5] = 0
    return xc

pow_2 = numpy.power(2, numpy.fliplr([numpy.arange(0, 15)])[0].astype(numpy.float64))

def val_transt(xi, k):
    """trasnlate values"""
    p2 = numpy.power(2, numpy.fliplr([numpy.arange(k-15, k)])[0].astype(numpy.float64)) # gets the powers of 2
    return (-1)**xi[:,0] * numpy.dot(xi[:, 1:], p2)

def val_transt1(xi, k):
    """trasnlate values"""
    p2 = numpy.power(2, numpy.fliplr([numpy.arange(k-15, k)])[0].astype(numpy.float64)) # gets the powers of 2
    return (-1)**xi[:,:,0] * numpy.dot(xi[:,:, 1:], p2)

def val_transf(xi, val):
    """transf vals new """
    return (-1)**xi[:,0] * numpy.dot(xi[:, 1:], pow_2) * val

def val_transf1(xi, val):
    return (-1)**xi[:,:,0] * numpy.dot(xi[:,:, 1:], pow_2) * val


def fit_prop_give_index(fitness):
    """
    fitness - array containing the fitness values of each stuff
    """
    fitness = numpy.abs(fitness)
    total_fitness = (fitness).sum()
    random_fitness = numpy.random.uniform(0, total_fitness)
    return numpy.argmax(fitness.cumsum() >= random_fitness)


def plot_GA(f, x, val,lim):
    plt.plot(f(val_transf1(x, val/2**15)))
    plt.ylim(ymax=lim)
    plt.show()


"----- GA -----------"

def mutation(x1, x2):
    """
    crossover & bitflip
    """
    co = 0.6
    N, n = x1.shape
    mp = 1/ n

    A = numpy.random.randint(1, n-2, N)
    B = numpy.rint(numpy.random.uniform(A+1, n, N)).astype(numpy.int)

    child1 = x1.copy()
    child2 = x2.copy()
    t = numpy.random.random(N) < co
    for i in range(N):
        if t[i]:
            child1[A[i]:B[i]] = x2[A[i]:B[i]]
            child2[A[i]:B[i]] = x1[A[i]:B[i]]

    Child1 = deepcopy(child1)
    Child1[numpy.random.random(child1.shape) <= mp] -= 1
    Child1 = numpy.abs(Child1)

    Child2 = deepcopy(child2)
    Child2[numpy.random.random(child2.shape) <= mp] -= 1
    Child2 = numpy.abs(Child2)
    if (numpy.abs(Child1) > 1).any():
        print("c1")
        print(Child1[numpy.abs(Child1).argmax()])
    if (numpy.abs(Child1) > 1).sum() > 0:
        print("a")
#    if (numpy.abs(Child2) > 1).any:
#        print(Child2)
#    done = 0
#    while not done:
#        c_o = numpy.abs(val_transt(child, k)) > val
#        if (c_o.sum() == 0):
#            done = 1
#        else:
#            child2 = child[c_o]
#            child2[numpy.random.random(child2.shape) < mp] -= 1
#            child[c_o] = numpy.abs(child2)
    return Child1, Child2


#def ga_init_vals(n, val, k):
#    """
#    n - nr of vars,
#    val - max value of var
#    k point at which "the binary goes to decimal"""
#    a
#    xi = numpy.zeros([100, n, 16])
#    for j in range(100):
#        for i in range(n):
#            done = 0
#            while not done:
#                xi[j, i] = binary_create()
#                if ((abs(val_transt(xi[j], k)) - val) <= 0 ).all():
#                    done = 1
#    return xi

def ga_init_vals(n):
    """new pop creation with improv"""
    x = numpy.random.random((100, n, 16))
    xi = x.copy()
    xi[x>0.5] = 1
    xi[x<=0.5] = 0
    return xi


def ga_cross(f, n, val):
    #make initial func vals
    max_val = f(numpy.ones((100, n)) * val)
    print(max_val)
    val /= 2**15
    x_old = ga_init_vals(n)
    f_val = f(val_transf1(x_old, val))

    iterations = 10**4
    results = numpy.zeros([iterations, n, 16]) #max fitness
    x_new = numpy.zeros_like(x_old)

    #scaling window
    scaling_w = numpy.ones(5) * max_val[0]
    scaling_w[-1] = f_val[f_val.argmax()]
    print(scaling_w)

    #first try with just 2 children
    for i in range(iterations):
        f_val = numpy.abs(f(val_transf1(x_old, val)))
        sct = deepcopy(scaling_w)
        scaling_w[:4] = sct[1:]
        scaling_w[-1] = f_val[f_val.argmax()]
        fitness = scaling_w[scaling_w.argmax()] - f_val
        if (fitness < 0).any():
            print(fitness[fitness.argmin()])
        elite = fitness.argmax() # index of the elite
        #print(fitness[elite], elite)
        tre = deepcopy(x_old[elite])
        for j in range(50):
            #remake the pop from old pop
            A = fit_prop_give_index(fitness) #so the closer you are to 0, the more chances there are
            B = fit_prop_give_index(fitness)

#            x_new[2*j] = mutation(x_old[A], x_old[B], k, val) #child1
#            x_new[2*j+1] = mutation(x_old[A], x_old[B], k, val) #child2
            x_new[2*j], x_new[2*j+1] = mutation(x_old[A], x_old[B])

        # print(results.shape, results[i].shape, elite, fitness.argmin(),fitness[fitness.argmin()] )
        results[i] = x_old[elite]

#        for j in range(100):
#            fitness[j] = f(val_transt(x_new[j], k))

        fitness_new = scaling_w[scaling_w.argmax()] - numpy.abs(f(val_transf1(x_new, val)))
        not_so_elite = fitness_new.argmin()
        x_new[not_so_elite] = tre
        x_old = deepcopy(x_new)
#        fitness_new = max_val - numpy.abs(f(val_transf1(x_new, val)))
#        print(i,val,fitness_new[not_so_elite],fitness[fitness.argmax()] > fitness_new[fitness_new.argmax()], fitness[fitness.argmax()], fitness_new[fitness_new.argmax()], fitness_new.argmax())

        if i % 100 == 0:
            print(i,fitness[fitness.argmax()] < fitness_new[fitness_new.argmax()], fitness[fitness.argmax()], fitness_new[fitness_new.argmax()], fitness_new.argmin())
            with open("{}.txt".format(f.__name__), 'a+') as dat:
                dat.write(numpy.str(results[i]) + "\n") # just in case

            del(dat)


    return results

    """for i in range(100000):
        A = np.random.randint(0, 100)
        B = np.random.randint(0, 100)

        if f(val_transt(x[A], k)) < f(val_transt(x[B], k)):
            parent1 = A
        else:
            parent1 = B

        A = np.random.randint(0, 100)
        B = np.random.randint(0, 100)

        if f(val_transt(x[A], k)) < f(val_transt(x[B], k)):
            parent2 = A
        else:
            parent2 = B

        child = mutation(x[parent1], x[parent2], k, val)
        child_fit = f(val_transt(child, k))

        if max_fit > child_fit:
            max_fit = child_fit

        A = np.random.randint(0, 100)
        B = np.random.randint(0, 100)

        if f(val_transt(x[A], k)) < f(val_transt(x[B], k)):
            x[B] = child
        else:
            x[A] = child

        # iter_no += 1
        results[i] = max_fit
        if i % 1000 == 0:
            print(i, max_fit)


    return results, child"""





"----- CCGA-----------"
def ccga_init_vals(n):
    """
    n - nr of vars,
    val - max value of var
    k point at which "the binary goes to decimal"""
    x = numpy.random.random([n, 100, 16])
    xi = deepcopy(x)
    xi[x > 0.5] = 1
    xi[x <= 0.5] = 0
    return xi


def ccga(f, n, val):
    max_val = f(numpy.ones((100, n)) * val)
    val = val / 2**15
    x_old = ccga_init_vals(n)
    f_val_init = numpy.zeros((n, 100))
    rand_dist = numpy.random.randint(0, 100, (n, 100, n))

    indv = numpy.zeros((n, 100, n, 16))
    for i in range(n):
        for j in range(100):
            rand_dist[i, j, i] = j
            for k in range(n):
                indv[i,j,k]  = x_old[i, rand_dist[i,j, k]]

    for i in range(n):
        f_val_init[i] = f(val_transf1(indv[i], val))
    #vals = f(val_transf1(rand_dist, val))
    return rand_dist, indv, f_val_init

z, aa, fv = ccga(rastrigin, 20, 5.12)
