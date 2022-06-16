from mef.iteration import IterationSet, show_results


cnn1d = IterationSet.load("logs/DeepSkyBlueLaura=cnn1d_shorts_128.pk")
m11_128 = IterationSet.load("logs/OldLaceHeather=m11_128.pk")
m11_256 = IterationSet.load("logs/OldLaceMichael=m11_256.pk")

show_results(cnn1d, m11_128)
show_results(m11_256, m11_128)
show_results(cnn1d, m11_256)

