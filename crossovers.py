import numpy as np
import itertools


def roll(G, S):
    Gv = []
    Gmeta = []
    for g in G:
        Gv.append(g.flatten())
        Gmeta.append(g.shape)

    Sv = []
    SMeta = []
    for sl in S:
        SSMeta = []
        for s in sl:
            Sv.append(s.flatten())
            SSMeta.append(s.shape)
        SMeta.append(SSMeta)

    GV = np.concatenate(Gv)
    SV = np.concatenate(Sv)
    v = np.concatenate((GV, SV))
    return v, (Gmeta, SMeta)


def unroll(vec, meta):
    Gmeta, SMeta = meta
    Gs = []
    start = 0
    for y, x in Gmeta:
        v = vec[start:start + y * x].reshape(y, x)
        Gs.append(v)
        start += y * x

    Ss = []
    for s in SMeta:
        Sss = []
        for y, x in s:
            v = vec[start:start + y * x].reshape(y, x)
            Sss.append(v)
            start += y * x
        Ss.append(Sss)
    return Gs, Ss


class RandomColumns():
    def __init__(self, meta, generator, problem):
        self.meta = meta
        self.generator = generator
        self.problem = problem

    def newG(self, G1, G2):
        vec = list(range(G1.shape[1]))
        np.random.shuffle(vec)
        v1 = vec[0:int(G1.shape[1] * 0.5)]
        v2 = vec[int(G1.shape[1] * 0.5):]

        newG = np.zeros(G1.shape)
        for x in v1:
            newG[:, x] = G1[:, x]
        for x in v2:
            newG[:, x] = G2[:, x]
        return newG, v1, v2

    def newS(self, S1, S2, v1, v2):
        newS = np.zeros(S1.shape)
        for x, y in list(itertools.product(v1, v1)):
            newS[x, y] = S1[x, y]
        for x, y in list(itertools.product(v2, v2)):
            newS[x, y] = S2[x, y]
        return newS

    def evolve(self, parents):
        newGs = []
        newSs = []
        G1, S1 = unroll(np.array(parents[0].variables), self.meta)
        G2, S2 = unroll(np.array(parents[1].variables), self.meta)
        for Gs1, Gs2, Ss1, Ss2 in zip(G1, G2, S1, S2):
            nG, v1, v2 = self.newG(Gs1, Gs2)
            newGs.append(nG)
            ns = []
            for s1, s2 in zip(Ss1, Ss2):
                nS = self.newS(s1, s2, v1, v2)
                ns.append(nS)
            newSs.append(ns)
        p = self.generator.generate(self.problem)
        p.variables, _ = roll(newGs, newSs)
        return [p]

# G1 = np.ones((10, 7))
# G2 = np.ones((10, 7))*0.5
#
# rc = RandomColumns()
# print(rc.newG(G1, G2))



# class ArithmeticCrossover:
#    def __init__(self, number_of_parents=2):
#        self.number_of_parents = number_of_parents
#
#    def cross(self, parents_list, w_normalized=None):
#        if len(parents_list) < self.number_of_parents:
#            raise RuntimeError("List of parents is to small.")
#
#        if w_normalized is None:
#            w = np.random.rand(self.number_of_parents)
#            w_normalized = w / np.sum(w)
#
#        Gs, Ss = zip(*parents_list)
#
#        # Gss = [x for x in zip(*Gs)]
#        Gc = []
#
#        for g in zip(*Gs):
#            gn = np.zeros(g[0].shape)
#            for i, gi in enumerate(g):
#                gn += w_normalized[i] * gi
#            Gc.append(gn)
#
#        Sc = []
#        for s in zip(*Ss):
#            Sc_ = []
#            for i in range(len(s[0])):
#                g = np.zeros(s[0][i].shape)
#                for j in range(len(s)):
#                    g += w_normalized[j] * s[j][i]
#                Sc_.append(g)
#            Sc.append(Sc_)
#        return Gc, Sc
