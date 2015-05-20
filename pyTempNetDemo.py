import pyTempNet as tn
import igraph 
import numpy as np
import matplotlib.pyplot as plt

##############################################################################
# Illustration of basic concept based on two simple examples
##############################################################################

# We can first define a simple temporal network in which all time-respecting 
# paths are equally likely
t1 = tn.TemporalNetwork()
t1.addEdge('a', 'c', 1)
t1.addEdge('c', 'e', 2)

t1.addEdge('b', 'c', 3)
t1.addEdge('c', 'd', 4)

t1.addEdge('b', 'c', 5)
t1.addEdge('c', 'e', 6)

t1.addEdge('a', 'c', 7)
t1.addEdge('c', 'd', 8)

# For illustration purposes, we can output a time-unfolded representation in tikz. 
# Simply compile the resulting LaTeX file to obtain a PDF figure. 
t1.exportTikzUnfolded('output/t1.tex')

# We can extract time-respecting paths of length two. If no different parameter delta 
# is given, two time-stamped edges (a,b,t) and (b,c,t') are considered to contribute 
# to a time-respecting path a -> b -> c iff 0 < t'-t <= delta = 1
t1.extractTwoPaths()

# We can now easily calculate the betweenness preference of individual nodes. Here, 
# the betweenness preference of node c should be zero (because in the example there is no 
# preference of node c to mediate time-respecting paths between any particular pair of nodes)
print('I^b(S;D) = ', tn.Measures.BetweennessPreference(t1,'c'))

# We can compute and plot the first-order aggregate network
g1 = t1.igraphFirstOrder()

visual_style = {}
visual_style["bbox"] = (600, 400)
visual_style["margin"] = 60
visual_style["vertex_size"] = 80
visual_style["vertex_label_size"] = 24
visual_style["vertex_color"] = "lightblue"
visual_style["edge_curved"] = 0.2
visual_style["edge_width"] = 1
visual_style["edge_arrow_size"] = 2

visual_style["layout"] = g1.layout_auto()
visual_style["vertex_label"] = g1.vs["name"]
visual_style["edge_label"] = g1.es["weight"]
igraph.plot(g1, 'output/t1_G1.pdf', **visual_style)

# We can also compute and plot the second-order aggregate network 
# corresponding to the temporal network
g2 = t1.igraphSecondOrder()
visual_style["layout"] = g2.layout_auto()
visual_style["vertex_label"] = g2.vs["name"]
visual_style["edge_label"] = g2.es["weight"]
igraph.plot(g2, 'output/t1_G2.pdf', **visual_style)


# Let us now consider a different example in which the statistics of time-respecting paths 
# *differs* from what we expect based on the time-aggregated network
t2 = tn.TemporalNetwork()
t2.addEdge('a', 'c', 1)
t2.addEdge('c', 'e', 2)

t2.addEdge('b', 'c', 3)
t2.addEdge('c', 'd', 4)

t2.addEdge('b', 'c', 5)
t2.addEdge('c', 'd', 6)

t2.addEdge('a', 'c', 7)
t2.addEdge('c', 'e', 8)

# Again, we can export the tikz-code for a time-unfolded representation of the temporal network
t2.exportTikzUnfolded('output/t2.tex')

# Again, we extract time-respecting paths of length two 
t2.extractTwoPaths()

# The first-order aggregate network is exactly the same like before, as it does not 
# capture the ordering of time-stamped links in the temporal network
g1 = t2.igraphFirstOrder()

visual_style["layout"] = g1.layout_auto()
visual_style["vertex_label"] = g1.vs["name"]
visual_style["edge_label"] = g1.es["weight"]
igraph.plot(g1, 'output/t2_G1.pdf', **visual_style)


# In this case we however have non-zero betweenness preference. Precisely, here node c 
# has a preference to mediate time-respecting paths between the pairs of nodes a and e as well as 
# b and d respectively. Furthermore, knowing the "source" of a time-respecting path through c 
# completely determines the target (source a determining target e and source b determining target d)
# Our uncertainity of two equally likely choices is reduced to one, which corresponds to a mutual information 
# of one bit. 
print('I^b(S;D) = ', tn.Measures.BetweennessPreference(t2,'c'))

# We can also output the corresponding (unnormalized) betweenness preference matrix. The entries provide us 
# with the number of different time-respecting paths through node c. The first row corresponds to node a, the second row 
# corresponds to node b, the first column corresponds to node e, the second column corresponds to node d. Here, the entries 
# reveal that there are two time-respecting paths a -> c -> e (first row) and two time-respecting paths b -> c -> d (second row)
# Tiem-respecting paths a -> c -> d and b -> c -> e (off-diagonal zero entries) are absent.
print('I^b(S;D) = ', tn.Measures.__BWPrefMatrix(t2,'c'))

# The changes in the statistics of time-respecting paths that are due to the reordering of the time-stamped edges is 
# captured by the fact that the second-order aggregate network is different from the one before (even though the first-order aggregate 
# network is the same!)
g2 = t2.igraphSecondOrder()

visual_style["layout"] = g2.layout_auto()
visual_style["vertex_label"] = g2.vs["name"]
visual_style["edge_label"] = g2.es["weight"]
igraph.plot(g2, 'output/t2_G2.pdf', **visual_style)

##############################################################################
# We now consider a larger (synthetic) example which we read from a TEDGE file
# We then demonstrate the spectral analysis of non-Markovian temporal networks
##############################################################################

# We can read so-called TEDGE files, which have the format
# > source, target, time
# i.e. they are simple (possibly unordered) lists of time-stamped links
# A header line naming individual columns should be included!
t = tn.TemporalNetwork.readFile('data/example.tedges', sep=' ')
print("Temporal network has", t.vcount(), "nodes")
print("Temporal network has", t.ecount(), "time-stamped edges")

# Let us extract all time-respecting paths of length two
t.extractTwoPaths()

# We can then plot the (first-order) time-aggregated representation
g1 = t.igraphFirstOrder()
visual_style = {}
visual_style["bbox"] = (600, 600)
visual_style["margin"] = 60
visual_style["edge_width"] = [x/100 for x in g1.es()["weight"]]
visual_style["vertex_size"] = 20
visual_style["vertex_label_size"] = 12
visual_style["vertex_color"] = "lightblue"
visual_style["vertex_label"] = g1.vs["name"]
visual_style["edge_arrow_size"] = 0.3
visual_style["layout"] = g1.layout_auto()

# Importantly, the weighted time-aggregated network does not show *any* particular structure
igraph.plot(g1, 'output/example_g1.pdf', **visual_style)

# Let us now plot the *expected* second-order time-aggregate network, which represents 
# the statistics of time-respecting paths expected at random. We see that this expected network 
# is *densely* connected
g2n = t.igraphSecondOrderNull()
visual_style = {}
visual_style["edge_width"] = [x/10000 for x in g2n.es()["weight"]]
visual_style["edge_arrow_size"] = 0.01
visual_style["vertex_color"] = "lightblue"
visual_style["vertex_size"] = 5
igraph.plot(g2n, 'output/example_g2n.pdf', **visual_style)

# Due to the specific *ordering* of time-stamped edges in the temporal network, the actual second-order aggregate network 
# can look completely different, meaning that only very specific time-respecting paths actually occur. Here, the second-order
# network shows three pronounced *temporal* communities that are not visible in the first-order network
g2 = t.igraphSecondOrder()
visual_style = {}
visual_style["edge_width"] = [x for x in g2.es()["weight"]]
visual_style["edge_arrow_size"] = 0.01
visual_style["vertex_color"] = "lightblue"
visual_style["vertex_size"] = 5
igraph.plot(g2, 'output/example_g2.pdf', **visual_style)

# The (visually) different second-order network shown above is a sign for the presence of *non-Markovian* characteristics
# which change causality in the temporal network. We can quantify the presence of these characteristics by calculating 
# the entropy growth rate ration of the second-order aggregate network. This value captures to what extend the actual 
# temporal network differs from a Markovian case, in which the next edge in a time-stamped edge sequence is independent from 
# the previous one. 

# Naturally, in this case we observe a strong difference expressed by an entropy growth rate ratio smaller than one
print("Entropy growth rate ratio is", tn.Measures.EntropyGrowthRateRatio(t))

# We can next study to what extent these characteristics influence dynamical processes on the temporal network. 
# Here, we consider a diffusion process, modeled by the convergence behavior of a random walk process on the temporal network. 
# With the next two functions, we can compute the time (i.e. number of random walk steps) required by a random walker to 
# converge. Precisely, we measure the average number of steps after which the total variation distance between the visitation
# probabilities and the stationary distribution is smaller than a given threshold of epsilon. 
speed_g2 = tn.Processes.RWDiffusion(t.igraphSecondOrder().components(mode="strong").giant(), epsilon=1e-4)
speed_g2n = tn.Processes.RWDiffusion(t.igraphSecondOrderNull().components(mode="strong").giant(), epsilon=1e-4)

# For small epsilon (i.e. large times t) we empirically observe that non-Markovian characteristics slow down the diffusion process
# by a factor of approx. 3.4 (values naturally differ in individual runs)
print("Empirical slow-down factor for diffusion is", speed_g2/speed_g2n)

# We can actually predict this analytically by means of a spectral analysis of the second-order time-aggregated network. 
# The following function analytically calculates the expected factor that is due to the changes in causality. 
# Here we expect diffusion in the temporal network to be slower by a factor of about 3.36 (compared to a Markovian temporal network 
# in which no order correlations change causality). This is well in line with our empirical observation above!
print("Analytical slow-down factor for diffusion is", tn.Measures.SlowDownFactor(t))


# Apart from predicting changes in diffusion speed, we can do some extended spectral analysis of the second-order network 
# based on a normalized Laplacian matrix. This analysis confirms that the actual temporal sequence has a smaller
# algebraic connectivity than a Markovian temporal network. This indicates that the *causal topology* of time-respecting 
# paths in the temporal network is less connected than expected at random, thus explaining the observed slow-down of diffusion. 
# This can intuitively be related to the presence of (temporal) community structures in the real temporal network (which we have seen 
# are absent in the null model)
print("Algebraic Connectivity (G2) =", tn.Measures.AlgebraicConn(t))
print("Algebraic Connectivity (G2 null) =", tn.Measures.AlgebraicConn(t, model="NULL"))

# Finally, we can apply some basic spectral partitioning of the temporal network based on the Fiedler vector, i.e. the eigenvector 
# corresponding to the second-smallest eigenvalue of the Laplacian matrix (i.e. algebraic connectivity). This allows us to 
# detect *temporal* communities. Importantly, here communities refer to *link communities*, as each entry of the Fiedler vector 
# in the second-order aggregate network corresponds to a *link* in the underlying temporal network.
# The distribution of entries in the Fiedler vector shows that links connect nodes in different 
# (temporal) communities.
fiedler = tn.Measures.FiedlerVector(t)

# Let us plot the entries in the vector (by index). We observe two *bands* of values that can be used for (recursive) spectral 
# partitioning
plt.clf()
plt.tick_params(axis='both', which='major', labelsize=20)
plt.tick_params(axis='both', which='minor', labelsize=20)       
plt.xlabel('$i$', fontsize=30)
plt.ylabel(r'$(v_2)_i$', fontsize=30)
plt.ylim(-.1, .1)
plt.subplots_adjust(bottom=0.25)
plt.subplots_adjust(left=0.25)
plt.scatter(range(len(fiedler)), np.real(fiedler), color="r")
plt.savefig("output/example_fiedler.pdf")
plt.close()

# The Fiedler vector of the second-order aggregate network corresponding to the null model does not show strong communities, 
# (as most values are clustered around zero)
fiedler_null = tn.Measures.FiedlerVector(t, model='NULL')

plt.clf()
plt.tick_params(axis='both', which='major', labelsize=20)
plt.tick_params(axis='both', which='minor', labelsize=20)       
plt.xlabel('$i$', fontsize=30)
plt.ylabel(r'$(v_2)_i$', fontsize=30)
plt.ylim(-.1, .1)
plt.subplots_adjust(bottom=0.25)
plt.subplots_adjust(left=0.25)
plt.scatter(range(len(fiedler_null)), np.real(fiedler_null), color="r")
plt.savefig("output/example_fiedler_null.pdf")
plt.close()

##############################################################################
# Demonstration using *real* time-stamped relational data
##############################################################################


# We now demonstrate the spectral analysis with some actual data. The first data set covers 
# E-Mail exchanges between  employees in a manufacturing company
t = tn.TemporalNetwork.readFile('data/manufacturing_30d_agg_3600_scc.tedges', sep=' ')
print("Temporal network has", t.vcount(), "nodes")
print("Temporal network has", t.ecount(), "time-stamped edges")

# The entropy growth rate ratio smaller than one confirms that the temporal network exhibits non-Markovian
# characteristics that are likely to change causality
print("Entropy growth rate ratio is", tn.Measures.EntropyGrowthRateRatio(t))

# Based on spectral properties, we analytically predict these characteristics to slow down diffusion 
# by a factor of about 3.2 (compared to a Markovian temporal network)
print("Analytical slow-down factor for diffusion is", tn.Measures.SlowDownFactor(t))

# We empirically confirm that this prediction is accurate
speed_g2 = tn.Processes.RWDiffusion(t.igraphSecondOrder().components(mode="strong").giant(), epsilon=1e-6)
speed_g2n = tn.Processes.RWDiffusion(t.igraphSecondOrderNull().components(mode="strong").giant(), epsilon=1e-6)
print("Empirical slow-down factor for diffusion is", speed_g2/speed_g2n)

# We next test a temporal network constructed from the Reality Mining data set 
t = tn.TemporalNetwork.readFile('data/RealityMining_agg_300s.tedges', sep=' ')
print("Temporal network has", t.vcount(), "nodes")
print("Temporal network has", t.ecount(), "time-stamped edges")

# Again, the temporal sequence deviates from a Markovian temporal network
print("Entropy growth rate ratio is", tn.Measures.EntropyGrowthRateRatio(t))

# Here, non-Markovian characteristics are expected to slow down diffusion by a factor of about ... 

# TODO: Returns wrong slow-down factor!
print("Slow-down factor for diffusion is", tn.Measures.SlowDownFactor(t))

# Let us again confirm this empirically ... 
speed_g2 = tn.Processes.RWDiffusion(t.igraphSecondOrder().components(mode="strong").giant(), epsilon=1e-6)
speed_g2n = tn.Processes.RWDiffusion(t.igraphSecondOrderNull().components(mode="strong").giant(), epsilon=1e-6)
print("Empirical slow-down factor for diffusion is", speed_g2/speed_g2n)

# Finally, we also find examples for temporal networks in which non-Markovian characteristics 
# result in a speed-up. For this, we consider a data set of time-stamped passenger flows in the 
# London Tube networkt = tn.TemporalNetwork.readFile('data/RealityMining_agg_300s_scc.tedges', sep=' ')

t = tn.TemporalNetwork.readFile('data/tube_flows_scc.tedges', sep=' ')
print("Temporal network has", t.vcount(), "nodes")
print("Temporal network has", t.ecount(), "time-stamped edges")

t.extractTwoPaths()
g2 = t.igraphSecondOrder().components(mode="STRONG").giant()

# TODO: This takes very long ... 
g2n = t.igraphSecondOrderNull().components(mode="STRONG").giant()

# Here, non-Markovian characteristics result in a speed-up of diffusion expressed by a slow-down factor 
# smaller than one
print("Entropy growth rate ratio is", tn.Measures.EntropyGrowthRateRatio(t))
print("Slow-down factor for diffusion is", tn.Measures.SlowDownFactor(t))


##############################################################################
# Demonstration using synthetic *trigram* files.
# Rather than containing time-stamped edges, from which time-respecting paths 
# are inferred, here we assume a file that directly contains the frequencies 
# of time-respecting paths of length two in the format
# > a,b,c,2
# which means the time-respecting path a -> b -> c occurred twice
##############################################################################

# In the following, we use synthetic data generated from a synthetic model that 
# demonstrates that non-Markovian characteristics in temporal networks can both 
# slow down and speed up dynamical processes 

# The first file corresponds to a case where non-Markovian properties *speed up*  a diffusion process
t_su = tn.TemporalNetwork.readFile('data/sigma0_75.trigram', fformat='TRIGRAM', sep = ' ')

# Again, the entropy growth rate ratio is smaller than one, verifying that the temporal network 
# has non-Markovian characteristics
print("Entropy growth rate ratio is", tn.Measures.EntropyGrowthRateRatio(t_su))

# In this case, we observe a slow-down factor *smaller than one* indicating a *speed up* 
# of diffusion
print("Analytical slow-down factor for diffusion is", tn.Measures.SlowDownFactor(t_su))

# Let us confirm that here diffusion is indeed faster!
speed_g2 = tn.Processes.RWDiffusion(t_su.igraphSecondOrder().components(mode="strong").giant(), epsilon=1e-4)
speed_g2n = tn.Processes.RWDiffusion(t_su.igraphSecondOrderNull().components(mode="strong").giant(), epsilon=1e-4)
print("Empirical slow-down factor for diffusion is", speed_g2/speed_g2n)

# The first-order aggregate network consists of two communities
g1 = t_su.igraphFirstOrder()
visual_style = {}
visual_style["edge_width"] = [np.sqrt(x) for x in g1.es()["weight"]]
visual_style["vertex_color"] = "lightblue"
visual_style["vertex_size"] = 10
visual_style["edge_arrow_size"] = 0.5
visual_style["layout"] = g1.layout_auto()
igraph.plot(g1, 'output/model_0_75_g1.pdf', **visual_style)

# The two communities are visible in the expected second-order network as well
g2n = t_su.igraphSecondOrderNull()
visual_style = {}
visual_style["edge_width"] = [np.sqrt(x) for x in g2n.es()["weight"]]
visual_style["edge_arrow_size"] = 0.01
visual_style["vertex_color"] = "lightblue"
visual_style["vertex_size"] = 5
visual_style["layout"] = g2n.layout_auto()
igraph.plot(g2n, 'output/model_0_75_g2n.pdf', **visual_style)

# In the actual second-order network of the temporal network, communities are (temporally) more 
# connected than expected at random, thus explaining the speed-up!
g2 = t_su.igraphSecondOrder()
visual_style["edge_width"] = [np.sqrt(x) for x in g2.es()["weight"]]
visual_style["edge_arrow_size"] = 0.01
visual_style["vertex_color"] = "lightblue"
visual_style["vertex_size"] = 5
igraph.plot(g2, 'output/model_0_75_g2.pdf', **visual_style)

# We can calculate algebraic connectivity in the second-order network. A comparison with the algebraic connectivity 
# of the (expected) Markovian temporal network shows that the temporal network is actually *better connected* than expected
print("Algebraic Connectivity (G2) =", tn.Measures.AlgebraicConn(t_su))
print("Algebraic Connectivity (G2 null) =", tn.Measures.AlgebraicConn(t_su, model="NULL"))

# Community structures are also visible in the Fiedler vector corresponding to the temporal network
fiedler = tn.Measures.FiedlerVector(t_su)

plt.clf()
plt.tick_params(axis='both', which='major', labelsize=20)
plt.tick_params(axis='both', which='minor', labelsize=20)       
plt.xlabel('$i$', fontsize=30)
plt.ylabel(r'$(v_2)_i$', fontsize=30)
plt.subplots_adjust(bottom=0.25)
plt.subplots_adjust(left=0.25)
plt.scatter(range(len(fiedler)), np.real(fiedler), color="r")
plt.savefig("output/model_0_75_fiedler.pdf")
plt.close()

# Interestingly, if we compare the Fiedler vector to that of the *expected* 
# second-order network, we see that community structures in the real temporal 
# network are not as strong as expected, i.e. non-Markovian properties *mitigate* 
# community structures
fiedler = tn.Measures.FiedlerVector(t_su, model="NULL")

plt.clf()
plt.tick_params(axis='both', which='major', labelsize=20)
plt.tick_params(axis='both', which='minor', labelsize=20)       
plt.xlabel('$i$', fontsize=30)
plt.ylabel(r'$(v_2)_i$', fontsize=30)
plt.subplots_adjust(bottom=0.25)
plt.subplots_adjust(left=0.25)
plt.scatter(range(len(fiedler)), np.real(fiedler), color="r")
plt.savefig("output/model_fiedler_null.pdf")
plt.close()


# The second file corresponds to a case where non-Markovian properties *slow down*  a diffusion process
t_sd = tn.TemporalNetwork.readFile('data/sigma-0_75.trigram', fformat='TRIGRAM', sep = ' ')

# Again, the entropy growth rate ratio is smaller than one, verifying that the temporal network 
# has indeed non-Markovian characteristics
print("Entropy growth rate ratio is", tn.Measures.EntropyGrowthRateRatio(t_sd))

# In this case, we observe a slow-down factor *larger than one* indicating a *slow down* 
# of diffusion (here by a factor of about 5.4)
print("Analytical slow-down factor for diffusion is", tn.Measures.SlowDownFactor(t_sd))

# We again confirm that here diffusion is indeed slower by a factor of more than five ... 
speed_g2 = tn.Processes.RWDiffusion(t_sd.igraphSecondOrder().components(mode="strong").giant(), epsilon=1e-4)
speed_g2n = tn.Processes.RWDiffusion(t_sd.igraphSecondOrderNull().components(mode="strong").giant(), epsilon=1e-4)
print("Empirical slow-down factor for diffusion is", speed_g2/speed_g2n)

# The algebraic connectivity in the second-order network is much smaller than before, which is due to the fact 
# that non-Markovian characteristics *enforce* the community structures. Compared to the algebraic connectivity 
# of the expected Markovian temporal network, here non-Markovian characteristics result in a less connected 
# *causal topology*
print("Algebraic Connectivity (G2) =", tn.Measures.AlgebraicConn(t_sd))

# This can also be seen in the distribution of entries of the Fiedler vector. The two value ranges 
# are much more separated, which means that community structures are stronger than before
# Furthermore, they are stronger than in the null model. 
fiedler = tn.Measures.FiedlerVector(t_sd)

plt.clf()
plt.tick_params(axis='both', which='major', labelsize=20)
plt.tick_params(axis='both', which='minor', labelsize=20)       
plt.xlabel('$i$', fontsize=30)
plt.ylabel(r'$(v_2)_i$', fontsize=30)
plt.subplots_adjust(bottom=0.25)
plt.subplots_adjust(left=0.25)
plt.scatter(range(len(fiedler)), fiedler, color="r")
plt.savefig("output/model_-0_75_fiedler.pdf")
plt.close()

# The expected second-order network is exactly the same as before (since the only difference between 
# the temporal networks is the ordering of time-stamped edges)
g2n = t_sd.igraphSecondOrderNull()
visual_style["edge_width"] = [np.sqrt(x) for x in g2n.es()["weight"]]
visual_style["edge_arrow_size"] = 0.01
visual_style["vertex_size"] = 5
igraph.plot(g2n, 'output/model_-0_75_g2n.pdf', **visual_style)

# The actual second-order aggregate network is different. Here we observe that compared to 
# the second-order network above, non-Markovian characteristics *enforce* community structures
g2 = t_sd.igraphSecondOrder()
visual_style["edge_width"] = [np.sqrt(x) for x in g2.es()["weight"]]
visual_style["vertex_size"] = 5
visual_style["edge_arrow_size"] = 0.01
igraph.plot(g2, 'output/model_-0_75_g2.pdf', **visual_style)

# As expected, the first-order network is exactly the same as before
g1 = t_sd.igraphFirstOrder()
visual_style = {}
visual_style["edge_width"] = [np.sqrt(x) for x in g1.es()["weight"]]
visual_style["vertex_color"] = "lightblue"
visual_style["vertex_size"] = 10
visual_style["edge_arrow_size"] = 0.5
visual_style["layout"] = g1.layout_auto()
igraph.plot(g1, 'output/model_-0_75_g1.pdf', **visual_style)