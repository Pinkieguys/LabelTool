import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import spam
from labeltool import segmentation
import matplotlib.pyplot as plt
import spam.kalisphera
import numpy


if __name__ == "__main__":
    # First create a seed image, this will be used to induce the segmentation problems
    seeds = spam.kalisphera.kalisphera.makeBlurryNoisySphere(
        (100, 100, 150),
        [[50, 50, 30], [50, 50, 50], [50, 50, 60], [50, 50, 90]],
        [15, 5, 5, 15],
    )
    # Label the seeds
    imSeedLab = spam.label.watershed(seeds > 0.5)
    # Create the binary of the "true" particles
    imGrey = spam.kalisphera.kalisphera.makeBlurryNoisySphere(
        (100, 100, 150),
        [[50, 50, 30], [50, 50, 60], [50, 50, 90], [50, 50, 120]],
        [15, 15, 15, 15],
    )
    # Labels the image and use the seeds, since there is an offset between the seeds and the true image, we will have segmentation problems
    imLabel = spam.label.watershed(imGrey > 0.5, markers=imSeedLab)
    # Let's take a look at it
    fig = plt.figure()
    plt.subplot(1, 1, 1)
    plt.gca().set_title("Label image")
    plt.imshow(imLabel[50, :, :], cmap=spam.label.randomCmap)
    plt.show()

    underSegCoeff = segmentation.detect_under_segmentation(imLabel)
    # Colour the label image with the coefficients
    imUnderSeg = spam.label.convertLabelToFloat(imLabel, underSegCoeff)
    # Let's plot them and see what do they look like
    fig = plt.figure()
    plt.subplot(1, 1, 1)
    plt.gca().set_title("Undersegmentation coefficient")
    plt.imshow(
        imUnderSeg[50, :, :],
        cmap=plt.cm.plasma,
        vmin=1,
        vmax=numpy.max(underSegCoeff),
    )
    plt.colorbar()
    plt.show()

    targetUnder = numpy.where(underSegCoeff > 1.2)[0]
    print(targetUnder)

    #
    imLabel2 = segmentation.fix_under_segmentation(imLabel, imGrey, targetUnder, None, imShowProgress=True,
                                               verbose=True, disableCoeffCheck=True)

    fig = plt.figure()
    plt.subplot(1, 1, 1)
    plt.gca().set_title("Label image")
    plt.imshow(imLabel2[50, :, :])
    plt.show()
