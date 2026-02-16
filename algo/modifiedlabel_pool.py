"""
这个函数库是spam的一个子函数库，主要是用来进行局部检测的，包括局部检测的组装，局部检测的单独检测，局部检测的可视化等等。
其原先在linux上运行，无法兼容Windows环境，经修正后可以在Windows环境下运行。
"""
import concurrent
import numpy as np
from spam.label.contacts import *
import multiprocessing
from spam.label.label import *
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from scipy.ndimage import binary_erosion
"""

"""
def start_process():
    return
def run_multi_process(func, args, startProcess=start_process):
    pool_size = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(
        processes=pool_size,
        initializer=startProcess,
        maxtasksperchild=2
    )

    # 使用进程池进行运算
    pool_outputs = pool.map(func, args)
    pool.close()
    pool.join()
    return pool_outputs

def localDetection_modiefied(subVolGrey,localThreshold,radiusThresh=None,subset=None, verbose=False):
    """
    local contact refinement
    checks whether two particles are in contact with a local threshold,
    that is higher than the global one used for binarisation

    Parameters
    ----------
        subVolGrey : 3D array
            Grey-scale volume

        localThreshold : integer or float, same type as the 3D array
            threshold for binarisation of the subvolume

        radiusThresh : integer, optional
            radius for excluding patches that might not be relevant,
            e.g. noise can lead to patches that are not connected with the grains in contact
            the patches are calculated with ``equivalentRadii()``
            Default is None and such patches are not excluded from the subvolume

    Returns
    -------
        CONTACT : boolean
            if True, that particles appear in contact for the local threshold

    Note
    ----
        see https://doi.org/10.1088/1361-6501/aa8dbf for further information
    """

    CONTACT = False

    if localThreshold < 0:
        #自适应
        try:
            strc = np.ones((5,5,5))
            mask1 = binary_erosion(subset['sub_1'],iterations=1,structure=strc)
            mask2 = binary_erosion(subset['sub_2'],iterations=1,structure=strc)
            k1=subVolGrey[mask1]
            k2=subVolGrey[mask2]
            localThreshold = np.min([np.min(k1),np.min(k2)]) + 1
        except:
            try:
                strc = np.ones((3, 3, 3))
                mask1 = binary_erosion(subset['sub_1'], iterations=1, structure=strc)
                mask2 = binary_erosion(subset['sub_2'], iterations=1, structure=strc)
                k1 = subVolGrey[mask1]
                k2 = subVolGrey[mask2]
                localThreshold = np.min([np.min(k1), np.min(k2)]) + 1
            except:
                mask1 = subset['sub_1'] > 0
                mask2 = subset['sub_2'] > 0
                k1 = subVolGrey[mask1]
                k2 = subVolGrey[mask2]
                localThreshold = np.min([np.min(k1), np.min(k2)]) + 1



    subVolBin = ((subVolGrey > localThreshold)*1).astype('uint8')
    if verbose:
        import toolusing
        toolusing.show(subVolBin,False)
    if radiusThresh is not None:
        # clean the image of isolated voxels or pairs of voxels
        # due to higher threshold they could exist
        subVolLab, numObjects = scipy.ndimage.measurements.label(subVolBin)
        if numObjects > 1:
            radii = spam.label.equivalentRadii( subVolLab )
            labelsToRemove = numpy.where( radii < radiusThresh )
            if len(labelsToRemove[0]) > 1:
                subVolLab = spam.label.removeLabels( subVolLab, labelsToRemove)
        subVolBin = ((subVolLab > 0)*1).astype('uint8')

    # fill holes
    subVolBin = scipy.ndimage.morphology.binary_fill_holes(subVolBin).astype('uint8')
    labeledArray, numObjects = scipy.ndimage.measurements.label(subVolBin, structure=None, output=None)
    if numObjects == 1:
        CONTACT = True

    return CONTACT
def fetchTwoGrains_more(volLab,labels,volGrey=None,boundingBoxes=None,padding=0,size_exclude=5):
    """
    Fetches the sub-volume of two grains from a labelled image

    Parameters
    ----------
        volLab : 3D array of integers
            Labelled volume, with lab.max() labels

        labels : 1x2 array of integers
            the two labels that should be contained in the subvolume
            
        volGrey : 3D array
            Grey-scale volume

        boundingBoxes : lab.max()x6 array of ints, optional
            Bounding boxes in format returned by ``boundingBoxes``.
            If not defined (Default = None), it is recomputed by running ``boundingBoxes``

        padding : integer
            padding of the subvolume
            for some purpose it might be benefitial to have a subvolume
            with a boundary of zeros

    Returns
    -------
        subVolLab : 3D array of integers
            labelled sub-volume containing the two input labels

        subVolBin : 3D array of integers
            binary subvolume

        subVolGrey : 3D array
            grey-scale subvolume
    """

    # check if bounding boxes are given
    if boundingBoxes is None:
        #print "bounding boxes are not given. calculating ...."
        boundingBoxes = spam.label.boundingBoxes(volLab)
    #else:
        #print "bounding boxes are given"
    lab1, lab2 = labels
    # Define output dictionary since we'll add different things to it
    output = {}
    # get coordinates of the big bounding box
    startZ = min( boundingBoxes[lab1,0], boundingBoxes[lab2,0] ) - padding
    stopZ  = max( boundingBoxes[lab1,1], boundingBoxes[lab2,1] ) + padding
    startY = min( boundingBoxes[lab1,2], boundingBoxes[lab2,2] ) - padding
    stopY  = max( boundingBoxes[lab1,3], boundingBoxes[lab2,3] ) + padding
    startX = min( boundingBoxes[lab1,4], boundingBoxes[lab2,4] ) - padding
    stopX  = max( boundingBoxes[lab1,5], boundingBoxes[lab2,5] ) + padding
    output['slice'] = (slice(startZ, stopZ+1),
                       slice(startY, stopY+1),
                       slice(startX, stopX+1))
    subVolLab = volLab[output['slice'][0].start:output['slice'][0].stop, 
                       output['slice'][1].start:output['slice'][1].stop, 
                       output['slice'][2].start:output['slice'][2].stop]

    subVolLab_A = numpy.where( subVolLab == lab1, lab1, 0 )
    subVolLab_B = numpy.where( subVolLab == lab2, lab2, 0 )
    subVolLab = subVolLab_A + subVolLab_B
    
    struc = numpy.zeros((3,3,3))
    struc[1,1:2,:]=1
    struc[1,:,1:2]=1
    struc[0,1,1]=1
    struc[2,1,1]=1
    subVolLab = spam.label.filterIsolatedCells(subVolLab, struc, size_exclude)
    output['subVolLab'] = subVolLab
    subVolBin = numpy.where( subVolLab != 0, 1, 0 )
    output['subVolBin'] = subVolBin

    output['sub_1'] = numpy.where( subVolLab == lab1, 1, 0 )
    output['sub_2'] = numpy.where( subVolLab == lab2, 1, 0 )

    if volGrey is not None:
        subVolGrey = volGrey[output['slice'][0].start:output['slice'][0].stop, 
                             output['slice'][1].start:output['slice'][1].stop, 
                             output['slice'][2].start:output['slice'][2].stop]
        subVolGrey = subVolGrey * subVolBin
        output['subVolGrey'] = subVolGrey
    
    return output

def funLocalDetectionAssembly(args):
    job, contactList, volLab, volGrey, localThreshold, radiusThresh, boundingBoxes = args
    grainA, grainB = contactList[job].astype("int")
    labels = [grainA, grainB]
    subset = fetchTwoGrains_more(volLab, labels, volGrey, boundingBoxes)
    contact = localDetection_modiefied(subset["subVolGrey"], localThreshold, radiusThresh,subset)
    if contact is True:
        return 1, grainA, grainB
    return 0,job

def funLocalDetectionAssembly_individually(args):
    import labelled_contacts_modiefied
    job, contactList, volLab, volGrey, localThreshold, radiusThresh, boundingBoxes, contact_volume = args
    grainA, grainB = contactList[job].astype("int")
    labels = [grainA, grainB]
    subset,inA,inB = labelled_contacts_modiefied.fetchTwoGrains_individually(volLab, labels, volGrey, boundingBoxes,contact_volume=contact_volume,contact_id=job+1)
    contact = localDetection(subset["subVolGrey"], localThreshold, radiusThresh)
    if contact is True:
        return 1, grainA, grainB, inA, inB
    return 0,job

def localDetectionAssembly_modiefied(
    volLab,
    volGrey,
    contactList,
    localThreshold,
    boundingBoxes=None,
    nProcesses=nProcessesDefault,
    radiusThresh=4,
    record_individually=False,
    contact_volume = None
):
    import progressbar

    # check if bounding boxes are given
    if boundingBoxes is None:
        # print "bounding boxes are not given. calculating ...."
        boundingBoxes = spam.label.boundingBoxes(volLab)

    contactListRefined = []
    numberOfJobs = len(contactList)
    label_to_remove = []
    # Create progressbar
    widgets = [
        progressbar.FormatLabel(""),
        " ",
        progressbar.Bar(),
        " ",
        progressbar.AdaptiveETA(),
    ]
    pbar = progressbar.ProgressBar(widgets=widgets, maxval=numberOfJobs)
    pbar.start()
    finishedNodes = 0

    if not record_individually:
        # Run multiprocessing
        print("并非逐一分析，不计算CIa，CIv")
        argslist = [(i, contactList, volLab, volGrey, localThreshold, radiusThresh, boundingBoxes) for i in
                    range(0, numberOfJobs)]
        if False:
            print("one by one")
            #returnsgroup = run_multi_process(funLocalDetectionAssembly,argslist)
            for job in [17746,17747]:
                args = argslist[job]
                returns = funLocalDetectionAssembly(args)
                if returns[0] == 1:
                    contactListRefined.append([returns[1], returns[2]])
                else:
                    #为什么这样写？？？？？不加注释真的变成未解之谜了
                    label_to_remove.append(returns[1] + 1)
        returnsgroup = run_multi_process(funLocalDetectionAssembly, argslist)
        for returns in returnsgroup:
            finishedNodes += 1
            widgets[0] = progressbar.FormatLabel("{}/{} ".format(finishedNodes, numberOfJobs))
            pbar.update(finishedNodes)
            if returns[0] == 1:
                contactListRefined.append([returns[1], returns[2]])
            else:
                #为什么这样写？？？？？不加注释真的变成未解之谜了
                label_to_remove.append(returns[1] + 1)
                #搞懂了，我没写错
                #这里的returns[1]是job，根据解读labelContacts的逻辑，job+1就是label

                

        # End progressbar
        pbar.finish()
        return numpy.asarray(contactListRefined),numpy.asarray(label_to_remove),1
    else:
        record_table = numpy.zeros((volLab.max() + 1, volLab.max() + 1))
        """
        record_table
        [i][j]位置记录的是颗粒i颗粒j接触面积中属于i的部分
        """
        returnsgroup = run_multi_process(funLocalDetectionAssembly_individually,
                                         [(i, contactList, volLab, volGrey, localThreshold, radiusThresh, boundingBoxes, contact_volume)
                                          for i in range(0, numberOfJobs)])
        for returns in returnsgroup:
            finishedNodes += 1
            widgets[0] = progressbar.FormatLabel("{}/{} ".format(finishedNodes, numberOfJobs))
            pbar.update(finishedNodes)
            if returns[0] == 1:
                contactListRefined.append([returns[1], returns[2]])
                grainA,grainB = returns[1], returns[2]
                inA,inB = returns[3], returns[4]
                record_table[grainA][grainB] = inA
                record_table[grainB][grainA] = inB
            else:
                label_to_remove.append(returns[1] + 1)
        # End progressbar
        pbar.finish()
        return numpy.asarray(contactListRefined), numpy.asarray(label_to_remove), numpy.asarray(record_table)



def fabricTensor_weight(orientations,weights):
    """
    Calculation of a second order fabric tensor from 3D unit vectors representing orientations

    Parameters
    ----------
        orientations: Nx3 array of floats
            Z, Y and X components of direction vectors
            Non-unit vectors are normalised.

    Returns
    -------
        N: 3x3 array of floats
            normalised second order fabric tensor
            with N[0,0] corresponding to z-z, N[1,1] to y-y and N[2,2] x-x

        F: 3x3 array of floats
            fabric tensor of the third kind (deviatoric part)
            with F[0,0] corresponding to z-z, F[1,1] to y-y and F[2,2] x-x

        a: float
            scalar anisotropy factor based on the deviatoric part F

    Note
    ----
        see [Kanatani, 1984] for more information on the fabric tensor
        and [Gu et al, 2017] for the scalar anisotropy factor

        Function contibuted by Max Wiebicke (Dresden University)
    """
    # from http://stackoverflow.com/questions/2850743/numpy-how-to-quickly-normalize-many-vectors
    norms = numpy.apply_along_axis(numpy.linalg.norm, 1, orientations)
    orientations = orientations / norms.reshape(-1, 1)
    # create an empty array
    N = numpy.zeros((3, 3))
    size = len(orientations)
    for i in range(size):
        orientation = orientations[i]
        tensProd = numpy.outer(orientation, orientation)
        N[:, :] = N[:, :] + tensProd*weights[i]
    # fabric tensor of the first kind
    N = N / (N[0,0] + N[1,1] + N[2,2])
    # fabric tensor of third kind
    F = (N - (numpy.trace(N) * (1. / 3.)) * numpy.eye(3, 3)) * (15. / 2.)
    # scalar anisotropy factor
    a = math.sqrt(3. / 2. * numpy.tensordot(F, F, axes=2))

    return N, F, a


def funContactOrientationsAssembly(args):
    job, contactList, volLab, volGrey, watershed, peakDistance, verbose,boundingBoxes = args
    grainA, grainB = contactList[job, 0:2].astype("int")
    labels = [grainA, grainB]
    subset = fetchTwoGrains(volLab, labels, volGrey, boundingBoxes)
    try:
        contactNormal, intervox, NotTreatedContact = spam.label.contactOrientations(
            subset["subVolBin"],
            subset["subVolLab"],
            watershed,
            peakDistance=peakDistance,
            verbose=verbose,
        )
    except:
        contactNormal, intervox, NotTreatedContact = [0,0,0],0,True
    return grainA, grainB, contactNormal[0], contactNormal[1], contactNormal[2], intervox, NotTreatedContact
def contactOrientationsAssembly_weight(
    volLab,
    volGrey,
    contactList,
    watershed="ITK",
    peakDistance=5,
    boundingBoxes=None,
    nProcesses=nProcessesDefault-4,
    verbose=False,
):
    print("contactOrientationsAssembly_weight使用nProcessesDefault-4")
    # check if bounding boxes are given
    if boundingBoxes is None:
        # print ("bounding boxes are not given. calculating ....")
        boundingBoxes = spam.label.boundingBoxes(volLab)

    contactOrientations = []
    numberOfJobs = len(contactList)
    nottreat = 0

    # Create progressbar
    widgets = [
        progressbar.FormatLabel(""),
        " ",
        progressbar.Bar(),
        " ",
        progressbar.AdaptiveETA(),
    ]
    pbar = progressbar.ProgressBar(widgets=widgets, maxval=numberOfJobs)
    pbar.start()
    finishedNodes = 0

    returnsgroup = run_multi_process(funContactOrientationsAssembly,[(i,contactList,volLab,volGrey,watershed,peakDistance,verbose,boundingBoxes) for i in range(0,numberOfJobs)])
    for returns in returnsgroup:
        finishedNodes += 1
        widgets[0] = progressbar.FormatLabel("{}/{} ".format(finishedNodes, numberOfJobs))
        pbar.update(finishedNodes)
        if returns is not None:
            nottreat += int(returns[6])
            """
            if returns[6]:
                continue
            """
            contactOrientations.append(
                [
                    returns[0],
                    returns[1],
                    returns[2],
                    returns[3],
                    returns[4],
                    returns[5]
                ]
            )
        else:
            raise Exception("还真有none的")

    pbar.finish()
    return numpy.asarray(contactOrientations),nottreat


def plotOrientations_weight(orientations_zyx, projection="lambert",VERBOSE = False, plot="both", binValueMin=None, binValueMax=None, binNormalisation=False, numberOfRings=9, pointMarkerSize=8, cmap=matplotlib.pyplot.cm.RdBu_r, title="", subtitle={"points":"", "bins":""}, saveFigPath=None,weights=None ):
    """
    Main function for plotting 3D orientations.
    This function plots orientations (described by unit-direction vectors) from a sphere onto a plane.

    One useful trick for evaluating these orientations is to project them with a "Lambert equal area projection", which means that an isotropic distribution of angles is projected as equally filling the projected space.

    Parameters
    ----------
        orientations : Nx3 numpy array of floats
            Z, Y and X components of direction vectors.
            Non-unit vectors are normalised.

        projection : string, optional
            Selects different projection modes:
                **lambert** : Equal-area projection, default and highly reccommended. See https://en.wikipedia.org/wiki/Lambert_azimuthal_equal-area_projection

                **equidistant** : equidistant projection

        plot : string, optional
            Selects which plots to show:
                **points** : shows projected points individually
                **bins** : shows binned orientations with counts inside each bin as colour
                **both** : shows both representations side-by-side, default

        title : string, optional
            Plot main title. Default = ""

        subtitle : dictionary, optional
            Sub-plot titles:
                **points** : Title for points plot. Default = ""
                **bins** : Title for bins plot. Default = ""

        binValueMin : int, optional
            Minimum colour-bar limits for bin view.
            Default = None (`i.e.`, auto-set)

        binValueMax : int, optional
            Maxmum colour-bar limits for bin view.
            Default = None (`i.e.`, auto-set)

        binNormalisation : bool, optional
            In binning mode, should bin counts be normalised by mean counts on all bins
            or absoulte counts?

        cmap : matplotlib colour map, optional
            Colourmap for number of counts in each bin in the bin view.
            Default = ``matplotlib.pyplot.cm.RdBu_r``

        numberOfRings : int, optional
            Number of rings (`i.e.`, radial bins) for the bin view.
            The other bins are set automatically to have uniform sized bins using an algorithm from Jacquet and Tabot.
            Default = 9 (quite small bins)

        pointMarkerSize : int, optional
            Size of points in point view (5 OK for many points, 25 good for few points/debugging).
            Default = 8 (quite big points)

        saveFigPath : string, optional
            Path to save figure to -- stops the graphical plotting.
            Default = None

    Returns
    -------
        None -- A matplotlib graph is created and show()n

    Note
    ----
        Authors: Edward Andò, Hugues Talbot, Clara Jacquet and Max Wiebicke
    """
    if weights == None:
        print("oops")
        return
    import matplotlib.pyplot

    # ========================================================================
    # ==== Reading in data, and formatting to x,y,z sphere                 ===
    # ========================================================================
    numberOfPoints = orientations_zyx.shape[0]

    # ========================================================================
    # ==== Check that all the vectors are unit vectors                     ===
    # ========================================================================
    if VERBOSE: print( "\t-> Normalising all vectors in x-y-z representation..." ),

    # from http://stackoverflow.com/questions/2850743/numpy-how-to-quickly-normalize-many-vectors
    norms = numpy.apply_along_axis( numpy.linalg.norm, 1, orientations_zyx )
    orientations_zyx = orientations_zyx / norms.reshape( -1, 1 )

    if VERBOSE: print( "done." )

    # ========================================================================
    # ==== At this point we should have clean x,y,z data in memory         ===
    # ========================================================================
    if VERBOSE: print( "\t-> We have %i orientations in memory."%( numberOfPoints ) )

    # Since this is the final number of vectors, at this point we can set up the
    #   matrices for the projection.
    projection_xy       = numpy.zeros( (numberOfPoints, 2) )

    # TODO: Check if there are any values less than zero or more that 2*pi
    projection_theta_r  = numpy.zeros( (numberOfPoints, 2) )

    # ========================================================================
    # ==== Projecting from x,y,z sphere to the desired projection          ===
    # ========================================================================
    # TODO: Vectorise this...
    for vectorN in range( numberOfPoints ):
        # unpack 3D x,y,z
        z,y,x = orientations_zyx[ vectorN ]
        #print "\t\txyz = ", x, y, z

        # fold over the negative half of the sphere
        #     flip every component of the vector over
        if z < 0: z = -z; y = -y; x = -x

        projection_xy[ vectorN ], projection_theta_r[ vectorN ] = spam.orientations.projectOrientation([z,y,x], "cartesian", projection)

    # get radiusMax based on projection
    #                                    This is only limited to sqrt(2) because we're flipping over the negative side of the sphere
    if projection == "lambert":         radiusMax = numpy.sqrt(2)
    elif projection == "stereo":        radiusMax = 1.0
    elif projection == "equidistant":   radiusMax = 1.0

    if VERBOSE: print( "\t-> Biggest projected radius (r,t) = {}".format( numpy.abs( projection_theta_r[:,1] ).max() ) )

    #print "projection_xy\n", projection_xy
    #print "\n\nprojection_theta_r\n", projection_theta_r


    if plot == "points" or plot == "both":
        fig = matplotlib.pyplot.figure()
        fig.suptitle( title )
        if plot == "both":
          ax  = fig.add_subplot( 121, polar=True )
        else:
          ax  = fig.add_subplot( 111, polar=True)

        ax.set_title( subtitle['points']+"\n" )

        # set the line along which the numbers are plotted to 0°
        #ax.set_rlabel_position(0)
        matplotlib.pyplot.axis( ( 0, math.pi*2, 0, radiusMax ) )

        # set radius grids to 15, 30, etc, which means 6 numbers (r=0 not included)
        radiusGridAngles = numpy.arange( 15, 91, 15 )
        radiusGridValues = []
        for angle in radiusGridAngles:
            #                        - project the 15, 30, 45 as spherical coords, and select the r part of theta r-
            #               - append to list of radii -

            radiusGridValues.append(spam.orientations.projectOrientation([0,angle*math.pi/180.0,1], "spherical", projection)[1][1])
        #                                       --- list comprehension to print 15°, 30°, 45° ----------
        ax.set_rgrids( radiusGridValues, labels=[ "%02i$^\circ$"%(x) for x in numpy.arange(  15,91,15) ], angle=None, fmt=None )
        ax.plot( projection_theta_r[:,0], projection_theta_r[:,1] , '.', markersize=pointMarkerSize )

        if plot == "points":
          matplotlib.pyplot.show()


    if plot == "bins" or plot == "both":
        # ========================================================================
        # ==== Binning the data -- this could be optional...                   ===
        # ========================================================================
        # This code inspired from Hugues Talbot and Clara Jaquet's developments.
        # As published in:
        #   Identifying and following particle-to-particle contacts in real granular media: an experimental challenge
        #   Gioacchino Viggiani, Edward Andò, Clara Jaquet and Hugues Talbot
        #   Keynote Lecture
        #   Particles and Grains 2013 Sydney
        #
        # ...The number of radial bins (numberOfRings)
        # defines the radial binning, and for each radial bin starting from the centre,
        # the number of angular bins is  4(2n + 1)
        #
        import matplotlib.patches
        #from matplotlib.colors import Normalize
        import matplotlib.colorbar
        import matplotlib.collections

        if plot == "both":
            ax  = fig.add_subplot( 122, polar=True )
        if plot == "bins":
            fig = matplotlib.pyplot.figure()
            ax  = fig.add_subplot( 111, polar=True)

        if VERBOSE: print( "\t-> Starting Data binning..." )

        # This must be an integer -- could well be a parameter if this becomes a function.
        if VERBOSE: print( "\t-> Number of Rings (radial bins) = ", numberOfRings )


        # As per the publication, the maximum number of bins for each ring, coming from the inside out is 4(2n + 1):
        numberOfAngularBinsPerRing = numpy.arange( 1, numberOfRings+1, 1 )
        numberOfAngularBinsPerRing = 4 * ( 2 * numberOfAngularBinsPerRing - 1 )

        if VERBOSE: print( "\t-> Number of angular bins per ring = ", numberOfAngularBinsPerRing )

        # defining an array with dimensions numberOfRings x numberOfAngularBinsPerRing
        binCounts = numpy.zeros( ( numberOfRings, numberOfAngularBinsPerRing[-1] ) )

        # ========================================================================
        # ==== Start counting the vectors into bins                            ===
        # ========================================================================
        for vectorN in range( numberOfPoints ):
            # unpack projected angle and radius for this point
            angle, radius = projection_theta_r[ vectorN, : ]
            www = weights[vectorN]

            # Flip over negative angles
            if angle < 0:             angle += 2*math.pi
            if angle > 2 * math.pi:   angle -= 2*math.pi

            # Calculate right ring number
            ringNumber = int(numpy.floor( radius / ( radiusMax / float(numberOfRings) ) ) )

            # Check for overflow
            if ringNumber > numberOfRings - 1:
                if VERBOSE: print( "\t-> Point with projected radius = %f is a problem (radiusMax = %f), putting in furthest  bin"%( radius, radiusMax ) )
                ringNumber = numberOfRings - 1

            # Calculate the angular bin
            angularBin = int( numpy.floor( ( angle ) / ( 2 * math.pi / float( numberOfAngularBinsPerRing[ ringNumber ] ) ) ) ) + 1

            #print "numberOfAngularBinsPerRing", numberOfAngularBinsPerRing[ringNumber] - 1
            # Check for overflow
            #  in case it doesn't belong in the last angularBin, it has to be put in the first one!
            if angularBin > numberOfAngularBinsPerRing[ringNumber] - 1:
                if VERBOSE: print( "\t-> Point with projected angle = %f does not belong to the last bin, putting in first bin"%( angle ) )
                angularBin = 0

            # now that we know what ring, and angular bin you're in add one count!
            binCounts[ ringNumber, angularBin ] += 1 * www

        # ========================================================================
        # === Plotting binned data                                             ===
        # ========================================================================

        plottingRadii = numpy.linspace( radiusMax/float(numberOfRings), radiusMax, numberOfRings )
        #print "Plotting radii:", plottingRadii

        #ax  = fig.add_subplot(122, polar=True)
        #matplotlib.pyplot.axis(  )
        #ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
        bars = []

        # add two fake, small circles at the beginning so that they are overwritten
        #   they will be coloured with the min and max colour
        #              theta   radius    width
        bars.append( [   0,   radiusMax,    2*math.pi ] )
        bars.append( [   0,   radiusMax,    2*math.pi ] )
        #bars.append( ax.bar( 0,   radiusMax,    2*math.pi, bottom=0.0 ) )
        #bars.append( ax.bar( 0,   radiusMax,    2*math.pi, bottom=0.0 ) )

        # --- flatifiying binned data for colouring wedges                    ===
        flatBinCounts = numpy.zeros( numpy.sum( numberOfAngularBinsPerRing ) + 2 )

        # Bin number as we go through the bins to add the counts in order to the flatBinCounts
        # This is two in order to skip the first to fake bins which set the colour bar.
        binNumber = 2

        # --- Plotting binned data, from the outside, inwards.                 ===
        if binNormalisation:
            avg_binCount = float(numberOfPoints)/numpy.sum( numberOfAngularBinsPerRing )
            #print "\t-> Number of points = ", numberOfPoints
            #print "\t-> Number of bins   = ", numpy.sum( numberOfAngularBinsPerRing )
            if VERBOSE: print( "\t-> Average binCount = ", avg_binCount )

        for ringNumber in range( numberOfRings )[::-1]:
            deltaTheta    = 360 / float( numberOfAngularBinsPerRing[ringNumber] )
            deltaThetaRad = 2 * math.pi / float( numberOfAngularBinsPerRing[ringNumber] )

            # --- Angular bins                                                 ---
            for angularBin in range( numberOfAngularBinsPerRing[ringNumber] ):
                # ...or add bars
                #                           theta                             radius                  width
                bars.append( [ angularBin*deltaThetaRad - deltaThetaRad/2.0, plottingRadii[ ringNumber ], deltaThetaRad ] )
                #bars.append( ax.bar( angularBin*deltaThetaRad - deltaThetaRad/2.0, plottingRadii[ ringNumber ], deltaThetaRad, bottom=0.0 ) )

                # Add the number of vectors counted for this bin
                if binNormalisation:
                    flatBinCounts[ binNumber ] = binCounts[ ringNumber, angularBin ]/avg_binCount
                else:
                    flatBinCounts[ binNumber ] = binCounts[ ringNumber, angularBin ]

                # Add one to bin number
                binNumber += 1

        del binNumber

        # figure out auto values if they're requested.
        if binValueMin is None: binValueMin = flatBinCounts[2::].min()
        if binValueMax is None: binValueMax = flatBinCounts[2::].max()

        # Add two flat values for the initial wedges.
        flatBinCounts[0] = binValueMin
        flatBinCounts[1] = binValueMax

        ##                           theta                   radius                          width
        barsPlot = ax.bar( numpy.array( bars )[:,0], numpy.array( bars )[:,1], width=numpy.array( bars )[:,2], bottom=0.0)

        for binCount,bar in zip(  flatBinCounts, barsPlot ):
            bar.set_facecolor( cmap(  ( binCount - binValueMin) / float( binValueMax - binValueMin ) ) )

        #matplotlib.pyplot.axis( [ 0, radiusMax, 0, radiusMax ] )
        matplotlib.pyplot.axis( [ 0, numpy.deg2rad(360), 0, radiusMax ] )

        #colorbar = matplotlib.pyplot.colorbar( barsPlot, norm=matplotlib.colors.Normalize( vmin=minBinValue, vmax=maxBinValue ) )
        # Set the colormap and norm to correspond to the data for which
        # the colorbar will be used.

        norm = matplotlib.colors.Normalize( vmin=binValueMin, vmax=binValueMax )

        # ColorbarBase derives from ScalarMappable and puts a colorbar
        # in a specified axes, so it has everything needed for a
        # standalone colorbar.  There are many more kwargs, but the
        # following gives a basic continuous colorbar with ticks
        # and labels.
        ax3 = fig.add_axes([0.9, 0.1, 0.03, 0.8])
        cb1 = matplotlib.colorbar.ColorbarBase( ax3, cmap=cmap, norm=norm )

        # set the line along which the numbers are plotted to 0°
        #ax.set_rlabel_position(0)

        # set radius grids to 15, 30, etc, which means 6 numbers (r=0 not included)
        radiusGridAngles = numpy.arange( 15, 91, 15 )
        radiusGridValues = []
        for angle in radiusGridAngles:
            #                        - project the 15, 30, 45 as spherical coords, and select the r part of theta r-
            #               - append to list of radii -
            radiusGridValues.append(spam.orientations.projectOrientation([0,angle*math.pi/180.0,1], "spherical", projection)[1][1])
        #                                       --- list comprehension to print 15°, 30°, 45° ----------
        ax.set_rgrids( radiusGridValues, labels=[ "%02i$^\circ$"%(x) for x in numpy.arange(  15,91,15) ], angle=None, fmt=None )

        fig.subplots_adjust(left=0.05,right=0.85)
        #cb1.set_label('Some Units')

        if saveFigPath is not None:
          matplotlib.pyplot.savefig( saveFigPath )
        else:
          matplotlib.pyplot.show()



def plotOrientations_ring(orientations_zyx, projection="lambert",VERBOSE = False, plot="both", binValueMin=None, binValueMax=None, binNormalisation=False, numberOfRings=9, pointMarkerSize=8, cmap=matplotlib.pyplot.cm.RdBu_r, title="", subtitle={"points":"", "bins":""}, saveFigPath=None,weights=None,excludeZero=True ):
    """
    Main function for plotting 3D orientations.
    This function plots orientations (described by unit-direction vectors) from a sphere onto a plane.

    One useful trick for evaluating these orientations is to project them with a "Lambert equal area projection", which means that an isotropic distribution of angles is projected as equally filling the projected space.

    Parameters
    ----------
        orientations : Nx3 numpy array of floats
            Z, Y and X components of direction vectors.
            Non-unit vectors are normalised.

        projection : string, optional
            Selects different projection modes:
                **lambert** : Equal-area projection, default and highly reccommended. See https://en.wikipedia.org/wiki/Lambert_azimuthal_equal-area_projection

                **equidistant** : equidistant projection

        plot : string, optional
            Selects which plots to show:
                **points** : shows projected points individually
                **bins** : shows binned orientations with counts inside each bin as colour
                **both** : shows both representations side-by-side, default

        title : string, optional
            Plot main title. Default = ""

        subtitle : dictionary, optional
            Sub-plot titles:
                **points** : Title for points plot. Default = ""
                **bins** : Title for bins plot. Default = ""

        binValueMin : int, optional
            Minimum colour-bar limits for bin view.
            Default = None (`i.e.`, auto-set)

        binValueMax : int, optional
            Maxmum colour-bar limits for bin view.
            Default = None (`i.e.`, auto-set)

        binNormalisation : bool, optional
            In binning mode, should bin counts be normalised by mean counts on all bins
            or absoulte counts?

        cmap : matplotlib colour map, optional
            Colourmap for number of counts in each bin in the bin view.
            Default = ``matplotlib.pyplot.cm.RdBu_r``

        numberOfRings : int, optional
            Number of rings (`i.e.`, radial bins) for the bin view.
            The other bins are set automatically to have uniform sized bins using an algorithm from Jacquet and Tabot.
            Default = 9 (quite small bins)

        pointMarkerSize : int, optional
            Size of points in point view (5 OK for many points, 25 good for few points/debugging).
            Default = 8 (quite big points)

        saveFigPath : string, optional
            Path to save figure to -- stops the graphical plotting.
            Default = None

    Returns
    -------
        None -- A matplotlib graph is created and show()n

    Note
    ----
        Authors: Edward Andò, Hugues Talbot, Clara Jacquet and Max Wiebicke
    """

    import matplotlib.pyplot

    # ========================================================================
    # ==== Reading in data, and formatting to x,y,z sphere                 ===
    # ========================================================================
    numberOfPoints = orientations_zyx.shape[0]

    # ========================================================================
    # ==== Check that all the vectors are unit vectors                     ===
    # ========================================================================
    if VERBOSE: print( "\t-> Normalising all vectors in x-y-z representation..." ),

    # from http://stackoverflow.com/questions/2850743/numpy-how-to-quickly-normalize-many-vectors
    norms = numpy.apply_along_axis( numpy.linalg.norm, 1, orientations_zyx )
    orientations_zyx = orientations_zyx / norms.reshape( -1, 1 )

    if VERBOSE: print( "done." )

    # ========================================================================
    # ==== At this point we should have clean x,y,z data in memory         ===
    # ========================================================================
    if VERBOSE: print( "\t-> We have %i orientations in memory."%( numberOfPoints ) )

    # Since this is the final number of vectors, at this point we can set up the
    #   matrices for the projection.
    projection_xy       = numpy.zeros( (numberOfPoints, 2) )

    # TODO: Check if there are any values less than zero or more that 2*pi
    projection_theta_r  = numpy.zeros( (numberOfPoints, 2) )

    # ========================================================================
    # ==== Projecting from x,y,z sphere to the desired projection          ===
    # ========================================================================
    # TODO: Vectorise this...
    for vectorN in range( numberOfPoints ):
        # unpack 3D x,y,z
        z,y,x = orientations_zyx[ vectorN ]
        #print "\t\txyz = ", x, y, z

        # fold over the negative half of the sphere
        #     flip every component of the vector over
        if z < 0: z = -z; y = -y; x = -x

        projection_xy[ vectorN ], projection_theta_r[ vectorN ] = spam.orientations.projectOrientation([z,y,x], "cartesian", projection)

    # get radiusMax based on projection
    #                                    This is only limited to sqrt(2) because we're flipping over the negative side of the sphere
    if projection == "lambert":         radiusMax = numpy.sqrt(2)
    elif projection == "stereo":        radiusMax = 1.0
    elif projection == "equidistant":   radiusMax = 1.0

    if VERBOSE: print( "\t-> Biggest projected radius (r,t) = {}".format( numpy.abs( projection_theta_r[:,1] ).max() ) )

    #print "projection_xy\n", projection_xy
    #print "\n\nprojection_theta_r\n", projection_theta_r


    if plot == "points" or plot == "both":
        fig = matplotlib.pyplot.figure()
        fig.suptitle( title )
        if plot == "both":
          ax  = fig.add_subplot( 121, polar=True )
        else:
          ax  = fig.add_subplot( 111, polar=True)

        ax.set_title( subtitle['points']+"\n" )

        # set the line along which the numbers are plotted to 0°
        #ax.set_rlabel_position(0)
        matplotlib.pyplot.axis( ( 0, math.pi*2, 0, radiusMax ) )

        # set radius grids to 15, 30, etc, which means 6 numbers (r=0 not included)
        radiusGridAngles = numpy.arange( 15, 91, 15 )
        radiusGridValues = []
        for angle in radiusGridAngles:
            #                        - project the 15, 30, 45 as spherical coords, and select the r part of theta r-
            #               - append to list of radii -

            radiusGridValues.append(spam.orientations.projectOrientation([0,angle*math.pi/180.0,1], "spherical", projection)[1][1])
        #                                       --- list comprehension to print 15°, 30°, 45° ----------
        ax.set_rgrids( radiusGridValues, labels=[ "%02i$^\circ$"%(x) for x in numpy.arange(  15,91,15) ], angle=None, fmt=None )
        ax.plot( projection_theta_r[:,0], projection_theta_r[:,1] , '.', markersize=pointMarkerSize )

        if plot == "points":
          matplotlib.pyplot.show()


    if plot == "bins" or plot == "both":
        # ========================================================================
        # ==== Binning the data -- this could be optional...                   ===
        # ========================================================================
        # This code inspired from Hugues Talbot and Clara Jaquet's developments.
        # As published in:
        #   Identifying and following particle-to-particle contacts in real granular media: an experimental challenge
        #   Gioacchino Viggiani, Edward Andò, Clara Jaquet and Hugues Talbot
        #   Keynote Lecture
        #   Particles and Grains 2013 Sydney
        #
        # ...The number of radial bins (numberOfRings)
        # defines the radial binning, and for each radial bin starting from the centre,
        # the number of angular bins is  4(2n + 1)
        #
        import matplotlib.patches
        #from matplotlib.colors import Normalize
        import matplotlib.colorbar
        import matplotlib.collections

        if plot == "both":
            ax  = fig.add_subplot( 122, polar=True )
        if plot == "bins":
            fig = matplotlib.pyplot.figure()
            ax  = fig.add_subplot( 111, polar=True)

        if VERBOSE: print( "\t-> Starting Data binning..." )

        # This must be an integer -- could well be a parameter if this becomes a function.
        if VERBOSE: print( "\t-> Number of Rings (radial bins) = ", numberOfRings )


        # As per the publication, the maximum number of bins for each ring, coming from the inside out is 4(2n + 1):
        numberOfAngularBinsPerRing = numpy.arange( 1, numberOfRings+1, 1 )
        numberOfAngularBinsPerRing = 4 * ( 2 * numberOfAngularBinsPerRing - 1 )

        if VERBOSE: print( "\t-> Number of angular bins per ring = ", numberOfAngularBinsPerRing )

        # defining an array with dimensions numberOfRings x numberOfAngularBinsPerRing
        binCounts = numpy.zeros( ( numberOfRings, numberOfAngularBinsPerRing[-1] ) )

        # ========================================================================
        # ==== Start counting the vectors into bins                            ===
        # ========================================================================
        for vectorN in range( numberOfPoints ):
            # unpack projected angle and radius for this point
            angle, radius = projection_theta_r[ vectorN, : ]

            # Flip over negative angles
            if angle < 0:             angle += 2*math.pi
            if angle > 2 * math.pi:   angle -= 2*math.pi

            # Calculate right ring number
            ringNumber = int(numpy.floor( radius / ( radiusMax / float(numberOfRings) ) ) )

            # Check for overflow
            if ringNumber > numberOfRings - 1:
                if VERBOSE: print( "\t-> Point with projected radius = %f is a problem (radiusMax = %f), putting in furthest  bin"%( radius, radiusMax ) )
                ringNumber = numberOfRings - 1

            # Calculate the angular bin
            angularBin = int( numpy.floor( ( angle ) / ( 2 * math.pi / float( numberOfAngularBinsPerRing[ ringNumber ] ) ) ) ) + 1

            #print "numberOfAngularBinsPerRing", numberOfAngularBinsPerRing[ringNumber] - 1
            # Check for overflow
            #  in case it doesn't belong in the last angularBin, it has to be put in the first one!
            if angularBin > numberOfAngularBinsPerRing[ringNumber] - 1:
                if VERBOSE: print( "\t-> Point with projected angle = %f does not belong to the last bin, putting in first bin"%( angle ) )
                angularBin = 0

            # now that we know what ring, and angular bin you're in add one count!
            binCounts[ ringNumber, angularBin ] += 1

        # ========================================================================
        # === Plotting binned data                                             ===
        # ========================================================================

        plottingRadii = numpy.linspace( radiusMax/float(numberOfRings), radiusMax, numberOfRings )
        #print "Plotting radii:", plottingRadii

        #ax  = fig.add_subplot(122, polar=True)
        #matplotlib.pyplot.axis(  )
        #ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
        bars = []

        # add two fake, small circles at the beginning so that they are overwritten
        #   they will be coloured with the min and max colour
        #              theta   radius    width
        bars.append( [   0,   radiusMax,    2*math.pi ] )
        bars.append( [   0,   radiusMax,    2*math.pi ] )
        #bars.append( ax.bar( 0,   radiusMax,    2*math.pi, bottom=0.0 ) )
        #bars.append( ax.bar( 0,   radiusMax,    2*math.pi, bottom=0.0 ) )

        # --- flatifiying binned data for colouring wedges                    ===
        flatBinCounts = numpy.zeros( numpy.sum( numberOfAngularBinsPerRing ) + 2 )

        # Bin number as we go through the bins to add the counts in order to the flatBinCounts
        # This is two in order to skip the first to fake bins which set the colour bar.
        binNumber = 2

        # --- Plotting binned data, from the outside, inwards.                 ===
        if binNormalisation:
            #avg_binCount = float(numberOfPoints)/numpy.sum( numberOfAngularBinsPerRing )
            avg_binCount = float(numberOfPoints)
            #print "\t-> Number of points = ", numberOfPoints
            #print "\t-> Number of bins   = ", numpy.sum( numberOfAngularBinsPerRing )
            if VERBOSE: print( "\t-> Average binCount = ", avg_binCount )

        for ringNumber in range( numberOfRings )[::-1]:
            deltaTheta    = 360 / float( numberOfAngularBinsPerRing[ringNumber] )
            deltaThetaRad = 2 * math.pi / float( numberOfAngularBinsPerRing[ringNumber] )

            # --- Angular bins                                                 ---
            for angularBin in range( numberOfAngularBinsPerRing[ringNumber] ):
                # ...or add bars
                #                           theta                             radius                  width
                bars.append( [ angularBin*deltaThetaRad - deltaThetaRad/2.0, plottingRadii[ ringNumber ], deltaThetaRad ] )
                #bars.append( ax.bar( angularBin*deltaThetaRad - deltaThetaRad/2.0, plottingRadii[ ringNumber ], deltaThetaRad, bottom=0.0 ) )

                # Add the number of vectors counted for this bin
                if binNormalisation:
                    flatBinCounts[ binNumber ] = binCounts[ ringNumber, angularBin ]/avg_binCount
                else:
                    flatBinCounts[ binNumber ] = binCounts[ ringNumber, angularBin ]

                # Add one to bin number
                binNumber += 1

        del binNumber

        # figure out auto values if they're requested.
        if binValueMin is None:
            fbc = flatBinCounts[2::]
            if excludeZero:
                binValueMin = fbc[np.nonzero(fbc)].min()
            else:
                binValueMin = fbc.min()
        if binValueMax is None: binValueMax = flatBinCounts[2::].max()

        # Add two flat values for the initial wedges.
        flatBinCounts[0] = binValueMin
        flatBinCounts[1] = binValueMax

        ##                           theta                   radius                          width
        barsPlot = ax.bar( numpy.array( bars )[:,0], numpy.array( bars )[:,1], width=numpy.array( bars )[:,2], bottom=0.0)

        for binCount,bar in zip(  flatBinCounts, barsPlot ):
            bar.set_facecolor( cmap(  ( binCount - binValueMin) / float( binValueMax - binValueMin ) ) )

        #matplotlib.pyplot.axis( [ 0, radiusMax, 0, radiusMax ] )
        matplotlib.pyplot.axis( [ 0, numpy.deg2rad(360), 0, radiusMax ] )

        #colorbar = matplotlib.pyplot.colorbar( barsPlot, norm=matplotlib.colors.Normalize( vmin=minBinValue, vmax=maxBinValue ) )
        # Set the colormap and norm to correspond to the data for which
        # the colorbar will be used.

        norm = matplotlib.colors.Normalize( vmin=binValueMin, vmax=binValueMax )

        # ColorbarBase derives from ScalarMappable and puts a colorbar
        # in a specified axes, so it has everything needed for a
        # standalone colorbar.  There are many more kwargs, but the
        # following gives a basic continuous colorbar with ticks
        # and labels.
        ax3 = fig.add_axes([0.9, 0.1, 0.03, 0.8])
        cb1 = matplotlib.colorbar.ColorbarBase( ax3, cmap=cmap, norm=norm )

        # set the line along which the numbers are plotted to 0°
        #ax.set_rlabel_position(0)

        # set radius grids to 15, 30, etc, which means 6 numbers (r=0 not included)
        radiusGridAngles = numpy.arange( 15, 91, 15 )
        radiusGridValues = []
        for angle in radiusGridAngles:
            #                        - project the 15, 30, 45 as spherical coords, and select the r part of theta r-
            #               - append to list of radii -
            radiusGridValues.append(spam.orientations.projectOrientation([0,angle*math.pi/180.0,1], "spherical", projection)[1][1])
        #                                       --- list comprehension to print 15°, 30°, 45° ----------
        ax.set_rgrids( radiusGridValues, labels=[ "%02i$^\circ$"%(x) for x in numpy.arange(  15,91,15) ], angle=None, fmt=None )

        fig.subplots_adjust(left=0.05,right=0.85)
        #cb1.set_label('Some Units')

        if saveFigPath is not None:
          matplotlib.pyplot.savefig( saveFigPath )
        else:
          matplotlib.pyplot.show()


def plotOrientations_modiefid(orientations_zyx, projection="lambert", plot="both", binValueMin=None, binValueMax=None, binNormalisation=False, numberOfRings=9, pointMarkerSize=8, cmap=matplotlib.pyplot.cm.RdBu_r, title="", subtitle={"points":"", "bins":""}, saveFigPath=None ):
    """
    Main function for plotting 3D orientations.
    This function plots orientations (described by unit-direction vectors) from a sphere onto a plane.

    One useful trick for evaluating these orientations is to project them with a "Lambert equal area projection", which means that an isotropic distribution of angles is projected as equally filling the projected space.

    Parameters
    ----------
        orientations : Nx3 numpy array of floats
            Z, Y and X components of direction vectors.
            Non-unit vectors are normalised.

        projection : string, optional
            Selects different projection modes:
                **lambert** : Equal-area projection, default and highly reccommended. See https://en.wikipedia.org/wiki/Lambert_azimuthal_equal-area_projection

                **equidistant** : equidistant projection

        plot : string, optional
            Selects which plots to show:
                **points** : shows projected points individually
                **bins** : shows binned orientations with counts inside each bin as colour
                **both** : shows both representations side-by-side, default

        title : string, optional
            Plot main title. Default = ""

        subtitle : dictionary, optional
            Sub-plot titles:
                **points** : Title for points plot. Default = ""
                **bins** : Title for bins plot. Default = ""

        binValueMin : int, optional
            Minimum colour-bar limits for bin view.
            Default = None (`i.e.`, auto-set)

        binValueMax : int, optional
            Maxmum colour-bar limits for bin view.
            Default = None (`i.e.`, auto-set)

        binNormalisation : bool, optional
            In binning mode, should bin counts be normalised by mean counts on all bins
            or absoulte counts?

        cmap : matplotlib colour map, optional
            Colourmap for number of counts in each bin in the bin view.
            Default = ``matplotlib.pyplot.cm.RdBu_r``

        numberOfRings : int, optional
            Number of rings (`i.e.`, radial bins) for the bin view.
            The other bins are set automatically to have uniform sized bins using an algorithm from Jacquet and Tabot.
            Default = 9 (quite small bins)

        pointMarkerSize : int, optional
            Size of points in point view (5 OK for many points, 25 good for few points/debugging).
            Default = 8 (quite big points)

        saveFigPath : string, optional
            Path to save figure to -- stops the graphical plotting.
            Default = None

    Returns
    -------
        None -- A matplotlib graph is created and show()n

    Note
    ----
        Authors: Edward Andò, Hugues Talbot, Clara Jacquet and Max Wiebicke
    """
    VERBOSE = False
    import matplotlib.pyplot
    matplotlib.rcParams['font.sans-serif'] = 'Arial'
    matplotlib.rcParams['font.size'] = 13
    # ========================================================================
    # ==== Reading in data, and formatting to x,y,z sphere                 ===
    # ========================================================================
    numberOfPoints = orientations_zyx.shape[0]

    # ========================================================================
    # ==== Check that all the vectors are unit vectors                     ===
    # ========================================================================
    if VERBOSE: print( "\t-> Normalising all vectors in x-y-z representation..." ),

    # from http://stackoverflow.com/questions/2850743/numpy-how-to-quickly-normalize-many-vectors
    norms = numpy.apply_along_axis( numpy.linalg.norm, 1, orientations_zyx )
    orientations_zyx = orientations_zyx / norms.reshape( -1, 1 )

    if VERBOSE: print( "done." )

    # ========================================================================
    # ==== At this point we should have clean x,y,z data in memory         ===
    # ========================================================================
    if VERBOSE: print( "\t-> We have %i orientations in memory."%( numberOfPoints ) )

    # Since this is the final number of vectors, at this point we can set up the
    #   matrices for the projection.
    projection_xy       = numpy.zeros( (numberOfPoints, 2) )

    # TODO: Check if there are any values less than zero or more that 2*pi
    projection_theta_r  = numpy.zeros( (numberOfPoints, 2) )

    # ========================================================================
    # ==== Projecting from x,y,z sphere to the desired projection          ===
    # ========================================================================
    # TODO: Vectorise this...
    for vectorN in range( numberOfPoints ):
        # unpack 3D x,y,z
        z,y,x = orientations_zyx[ vectorN ]
        #print "\t\txyz = ", x, y, z

        # fold over the negative half of the sphere
        #     flip every component of the vector over
        if z < 0: z = -z; y = -y; x = -x

        projection_xy[ vectorN ], projection_theta_r[ vectorN ] = spam.orientations.projectOrientation([z,y,x], "cartesian", projection)

    # get radiusMax based on projection
    #                                    This is only limited to sqrt(2) because we're flipping over the negative side of the sphere
    if projection == "lambert":         radiusMax = numpy.sqrt(2)
    elif projection == "stereo":        radiusMax = 1.0
    elif projection == "equidistant":   radiusMax = 1.0

    if VERBOSE: print( "\t-> Biggest projected radius (r,t) = {}".format( numpy.abs( projection_theta_r[:,1] ).max() ) )

    #print "projection_xy\n", projection_xy
    #print "\n\nprojection_theta_r\n", projection_theta_r


    if plot == "points" or plot == "both":
        fig = matplotlib.pyplot.figure()
        fig.suptitle( title )
        if plot == "both":
          ax  = fig.add_subplot( 121, polar=True )
        else:
          ax  = fig.add_subplot( 111, polar=True)

        ax.set_title( subtitle['points']+"\n" )

        # set the line along which the numbers are plotted to 0°
        #ax.set_rlabel_position(0)
        matplotlib.pyplot.axis( ( 0, math.pi*2, 0, radiusMax ) )

        # set radius grids to 15, 30, etc, which means 6 numbers (r=0 not included)
        radiusGridAngles = numpy.arange( 15, 91, 15 )
        radiusGridValues = []
        for angle in radiusGridAngles:
            #                        - project the 15, 30, 45 as spherical coords, and select the r part of theta r-
            #               - append to list of radii -

            radiusGridValues.append(spam.orientations.projectOrientation([0,angle*math.pi/180.0,1], "spherical", projection)[1][1])
        #                                       --- list comprehension to print 15°, 30°, 45° ----------
        ax.set_rgrids( radiusGridValues, labels=[ "%02i$^\circ$"%(x) for x in numpy.arange(  15,91,15) ], angle=None, fmt=None )
        ax.plot( projection_theta_r[:,0], projection_theta_r[:,1] , '.', markersize=pointMarkerSize )

        if plot == "points":
          matplotlib.pyplot.show()


    if plot == "bins" or plot == "both":
        # ========================================================================
        # ==== Binning the data -- this could be optional...                   ===
        # ========================================================================
        # This code inspired from Hugues Talbot and Clara Jaquet's developments.
        # As published in:
        #   Identifying and following particle-to-particle contacts in real granular media: an experimental challenge
        #   Gioacchino Viggiani, Edward Andò, Clara Jaquet and Hugues Talbot
        #   Keynote Lecture
        #   Particles and Grains 2013 Sydney
        #
        # ...The number of radial bins (numberOfRings)
        # defines the radial binning, and for each radial bin starting from the centre,
        # the number of angular bins is  4(2n + 1)
        #
        import matplotlib.patches
        #from matplotlib.colors import Normalize
        import matplotlib.colorbar
        import matplotlib.collections

        if plot == "both":
            ax  = fig.add_subplot( 122, polar=True )
        if plot == "bins":
            fig = matplotlib.pyplot.figure()
            ax  = fig.add_subplot( 111, polar=True)

        if VERBOSE: print( "\t-> Starting Data binning..." )

        # This must be an integer -- could well be a parameter if this becomes a function.
        if VERBOSE: print( "\t-> Number of Rings (radial bins) = ", numberOfRings )


        # As per the publication, the maximum number of bins for each ring, coming from the inside out is 4(2n + 1):
        numberOfAngularBinsPerRing = numpy.arange( 1, numberOfRings+1, 1 )
        numberOfAngularBinsPerRing = 4 * ( 2 * numberOfAngularBinsPerRing - 1 )

        if VERBOSE: print( "\t-> Number of angular bins per ring = ", numberOfAngularBinsPerRing )

        # defining an array with dimensions numberOfRings x numberOfAngularBinsPerRing
        binCounts = numpy.zeros( ( numberOfRings, numberOfAngularBinsPerRing[-1] ) )

        # ========================================================================
        # ==== Start counting the vectors into bins                            ===
        # ========================================================================
        for vectorN in range( numberOfPoints ):
            # unpack projected angle and radius for this point
            angle, radius = projection_theta_r[ vectorN, : ]

            # Flip over negative angles
            if angle < 0:             angle += 2*math.pi
            if angle > 2 * math.pi:   angle -= 2*math.pi

            # Calculate right ring number
            ringNumber = int(numpy.floor( radius / ( radiusMax / float(numberOfRings) ) ) )

            # Check for overflow
            if ringNumber > numberOfRings - 1:
                if VERBOSE: print( "\t-> Point with projected radius = %f is a problem (radiusMax = %f), putting in furthest  bin"%( radius, radiusMax ) )
                ringNumber = numberOfRings - 1

            # Calculate the angular bin
            angularBin = int( numpy.floor( ( angle ) / ( 2 * math.pi / float( numberOfAngularBinsPerRing[ ringNumber ] ) ) ) ) + 1

            #print "numberOfAngularBinsPerRing", numberOfAngularBinsPerRing[ringNumber] - 1
            # Check for overflow
            #  in case it doesn't belong in the last angularBin, it has to be put in the first one!
            if angularBin > numberOfAngularBinsPerRing[ringNumber] - 1:
                if VERBOSE: print( "\t-> Point with projected angle = %f does not belong to the last bin, putting in first bin"%( angle ) )
                angularBin = 0

            # now that we know what ring, and angular bin you're in add one count!
            binCounts[ ringNumber, angularBin ] += 1

        # ========================================================================
        # === Plotting binned data                                             ===
        # ========================================================================

        plottingRadii = numpy.linspace( radiusMax/float(numberOfRings), radiusMax, numberOfRings )
        #print "Plotting radii:", plottingRadii

        #ax  = fig.add_subplot(122, polar=True)
        #matplotlib.pyplot.axis(  )
        #ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
        bars = []

        # add two fake, small circles at the beginning so that they are overwritten
        #   they will be coloured with the min and max colour
        #              theta   radius    width
        bars.append( [   0,   radiusMax,    2*math.pi ] )
        bars.append( [   0,   radiusMax,    2*math.pi ] )
        #bars.append( ax.bar( 0,   radiusMax,    2*math.pi, bottom=0.0 ) )
        #bars.append( ax.bar( 0,   radiusMax,    2*math.pi, bottom=0.0 ) )

        # --- flatifiying binned data for colouring wedges                    ===
        flatBinCounts = numpy.zeros( numpy.sum( numberOfAngularBinsPerRing ) + 2 )

        # Bin number as we go through the bins to add the counts in order to the flatBinCounts
        # This is two in order to skip the first to fake bins which set the colour bar.
        binNumber = 2

        # --- Plotting binned data, from the outside, inwards.                 ===
        if binNormalisation:
            #avg_binCount = float(numberOfPoints)/numpy.sum( numberOfAngularBinsPerRing )
            avg_binCount = float(numberOfPoints)
            #print "\t-> Number of points = ", numberOfPoints
            #print "\t-> Number of bins   = ", numpy.sum( numberOfAngularBinsPerRing )
            if VERBOSE: print( "\t-> Average binCount = ", avg_binCount )

        for ringNumber in range( numberOfRings )[::-1]:
            deltaTheta    = 360 / float( numberOfAngularBinsPerRing[ringNumber] )
            deltaThetaRad = 2 * math.pi / float( numberOfAngularBinsPerRing[ringNumber] )

            # --- Angular bins                                                 ---
            for angularBin in range( numberOfAngularBinsPerRing[ringNumber] ):
                # ...or add bars
                #                           theta                             radius                  width
                bars.append( [ angularBin*deltaThetaRad - deltaThetaRad/2.0, plottingRadii[ ringNumber ], deltaThetaRad ] )
                #bars.append( ax.bar( angularBin*deltaThetaRad - deltaThetaRad/2.0, plottingRadii[ ringNumber ], deltaThetaRad, bottom=0.0 ) )

                # Add the number of vectors counted for this bin
                flatBinCounts[binNumber] = binCounts[ringNumber, angularBin]
                # Add one to bin number
                binNumber += 1

        del binNumber
        if binNormalisation:
            flatBinCounts = flatBinCounts / flatBinCounts[2::].max()

        # figure out auto values if they're requested.
        if binValueMin is None: binValueMin = flatBinCounts[2::].min()
        if binValueMax is None: binValueMax = flatBinCounts[2::].max()

        # Add two flat values for the initial wedges.
        flatBinCounts[0] = binValueMin
        flatBinCounts[1] = binValueMax



        ##                           theta                   radius                          width
        barsPlot = ax.bar( numpy.array( bars )[:,0], numpy.array( bars )[:,1], width=numpy.array( bars )[:,2], bottom=0.0)

        for binCount,bar in zip(  flatBinCounts, barsPlot ):
            bar.set_facecolor( cmap(  ( binCount - binValueMin) / float( binValueMax - binValueMin ) ) )

        #matplotlib.pyplot.axis( [ 0, radiusMax, 0, radiusMax ] )
        matplotlib.pyplot.axis( [ 0, numpy.deg2rad(360), 0, radiusMax ] )

        #colorbar = matplotlib.pyplot.colorbar( barsPlot, norm=matplotlib.colors.Normalize( vmin=minBinValue, vmax=maxBinValue ) )
        # Set the colormap and norm to correspond to the data for which
        # the colorbar will be used.

        norm = matplotlib.colors.Normalize( vmin=binValueMin, vmax=binValueMax )

        # ColorbarBase derives from ScalarMappable and puts a colorbar
        # in a specified axes, so it has everything needed for a
        # standalone colorbar.  There are many more kwargs, but the
        # following gives a basic continuous colorbar with ticks
        # and labels.
        #这是一种安排colorbar图例的方式，与三维直方图的不同。两者当中有一个更高明
        #ax3 = fig.add_axes([0.9, 0.1, 0.03*1.1, 0.8*1.1])
        #cb1 = matplotlib.colorbar.ColorbarBase( ax3, cmap=cmap, norm=norm )

        # set the line along which the numbers are plotted to 0°
        #ax.set_rlabel_position(0)

        # set radius grids to 15, 30, etc, which means 6 numbers (r=0 not included)
        radiusGridAngles = numpy.arange( 15, 91, 15 )
        radiusGridValues = []
        for angle in radiusGridAngles:
            #                        - project the 15, 30, 45 as spherical coords, and select the r part of theta r-
            #               - append to list of radii -
            radiusGridValues.append(spam.orientations.projectOrientation([0,angle*math.pi/180.0,1], "spherical", projection)[1][1])
        #                                       --- list comprehension to print 15°, 30°, 45° ----------
        ax.set_rgrids( radiusGridValues, labels=[ "%02i$^\circ$"%(x) for x in numpy.arange(  15,91,15) ], angle=None, fmt=None )

        fig.subplots_adjust(left=0.05,right=0.85)
        #cb1.set_label('Some Units')

        if saveFigPath is not None:
            #ax.figure.set_size_inches(4.8 / 2.54, 4.8 / 2.54)
            matplotlib.pyplot.savefig( saveFigPath,bbox_inches='tight')
        else:
            matplotlib.pyplot.show()
        plt.clf()
        return binValueMin,binValueMax



def computeAngle(args):
    i, data, icoVectors = args
    # Get the orientation vector
    orientationVect = data[i]
    # Exchange Z and X position - for plotting
    orientationVect = [orientationVect[2], orientationVect[1], orientationVect[0]]
    # Create the result array
    angle = []
    for i in range(len(icoVectors)):
        # Compute the angle between them
        angle.append(numpy.arccos(numpy.clip(numpy.dot(orientationVect, icoVectors[i]), -1, 1)))
    # Get the index
    minIndex = numpy.argmin(angle)
    return minIndex
def plotSphericalHistogram_modiefied(orientations, subDiv=3, reflection=True, maxVal=None, verbose=True, color=None, viewAnglesDeg=[25, 45], title=None, saveFigPath=None):
    import spam.orientations
    matplotlib.rcParams['font.sans-serif'] = 'Arial'
    matplotlib.rcParams['font.size'] = 14

    # Internal function for binning data into the icosphere faces

    def binIcosphere(data, icoVectors, verbose):
        # Create counts array
        counts = numpy.zeros(len(icoVectors))

        # Create progressbar
        if verbose:
            widgets = [progressbar.FormatLabel(""), " ", progressbar.Bar(), " ", progressbar.AdaptiveETA()]
            pbar = progressbar.ProgressBar(widgets=widgets, maxval=len(data))
            pbar.start()
        finishedOrientations = 0

        # Run multiprocessing
        # TODO:开销很大，data和icoVectors应该以某种共用方式定义，使用globalVar或multiprocessing提供的shared_list操作
        returns = run_multi_process(computeAngle,[(i,data,icoVectors) for i in range(len(data))])
        for singlereturn in returns:
            finishedOrientations += 1
            if verbose:
                widgets[0] = progressbar.FormatLabel("{}/{} ".format(finishedOrientations, len(data)))
                pbar.update(finishedOrientations)
            index = singlereturn
            counts[index] += 1

        return counts

    # Get number of points
    orientations.shape[0]
    # Check that they are 3D vectors
    if orientations.shape[1] != 3:
        print("\nspam.helpers.orientationPlotter.plotSphericalHistogram: The input vectors are not 3D")
        return
    # from http://stackoverflow.com/questions/2850743/numpy-how-to-quickly-normalize-many-vectors
    norms = numpy.apply_along_axis(numpy.linalg.norm, 1, orientations)
    orientations = orientations / norms.reshape(-1, 1)
    # Check if we can reflect the vectors
    if reflection:
        orientations = numpy.vstack([orientations, -1 * orientations])
    # Create the icosphere
    if verbose:
        print("\nspam.helpers.orientationPlotter.plotSphericalHistogram: Creating the icosphere")
    icoVerts, icoFaces, icoVectors = spam.orientations.generateIcosphere(subDiv)
    # Bin the data
    if verbose:
        print("\nspam.helpers.orientationPlotter.plotSphericalHistogram: Binning the data")
    counts = binIcosphere(orientations, icoVectors, verbose=verbose)
    # Now we are ready to plot
    if verbose:
        print("\nspam.helpers.orientationPlotter.plotSphericalHistogram: Plotting")

    # Create the figure
    fig = matplotlib.pyplot.figure()
    # ax = fig.gca(projection='3d')
    ax = fig.add_subplot(projection="3d")

    if color is None:
        cmap = matplotlib.pyplot.cm.RdBu_r
    else:
        cmap = color
    norm = matplotlib.pyplot.Normalize(vmin=0, vmax=1)
    if maxVal is None:
        maxVal = numpy.max(counts)

    # Don't do it like this, make empty arrays!
    # points = []
    # connectivityMatrix = []

    # Loop through each of the faces
    for i in range(len(icoFaces)):
        # Get the corresponding radius
        radii = counts[i] / maxVal
        if radii != 0:
            # Get the face
            face = icoFaces[i]
            # Get the vertices
            P1 = numpy.asarray(icoVerts[face[0]])
            P2 = numpy.asarray(icoVerts[face[1]])
            P3 = numpy.asarray(icoVerts[face[2]])
            # Extend the vertices as needed by the radius
            P1 = radii * P1 / numpy.linalg.norm(P1)
            P2 = radii * P2 / numpy.linalg.norm(P2)
            P3 = radii * P3 / numpy.linalg.norm(P3)
            # Combine the vertices
            vertices = numpy.asarray([numpy.array([0, 0, 0]), P1, P2, P3])

            # for vertex in vertices:
            # points.append(vertex)
            # connectivityMatrix.append([len(points)-1, len(points)-2, len(points)-3, len(points)-4])

            # Add the points to the scatter3D
            ax.scatter3D(vertices[:, 0], vertices[:, 1], vertices[:, 2], s=0)
            # Create each face
            face1 = numpy.array([vertices[0], vertices[1], vertices[2]])
            face2 = numpy.array([vertices[0], vertices[1], vertices[3]])
            face3 = numpy.array([vertices[0], vertices[3], vertices[2]])
            face4 = numpy.array([vertices[3], vertices[1], vertices[2]])

            # Plot each face!
            ax.add_collection3d(Poly3DCollection([face1, face2, face3, face4], facecolors=cmap(norm(radii)), linewidths=0.5, edgecolors="k"))

    # import spam.helpers
    # spam.helpers.writeUnstructuredVTK(numpy.array(points), numpy.array(connectivityMatrix), cellData={'counts': counts})

    # Extra parameters for the axis
    ax.set_box_aspect([1, 1, 1])
    matplotlib.pyplot.xlim(-1.1, 1.1)
    matplotlib.pyplot.ylim(-1.1, 1.1)
    ax.set_zlim(-1.1, 1.1)
    ax.view_init(viewAnglesDeg[0], viewAnglesDeg[1])
    # Set the colorbar

    norm = matplotlib.colors.Normalize(vmin=0, vmax=maxVal)
    sm = matplotlib.pyplot.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    matplotlib.pyplot.colorbar(sm,shrink=1.1)

    hideAxes = True
    if hideAxes:
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.zaxis.set_ticklabels([])
        # Remove the ticks labels and lines
        ax = matplotlib.pyplot.gca()
        for line in ax.xaxis.get_ticklines():
            line.set_visible(False)
        for line in ax.yaxis.get_ticklines():
            line.set_visible(False)
        for line in ax.zaxis.get_ticklines():
            line.set_visible(False)
            # 去掉网格线
        ax.grid(False)

        # 去掉坐标轴
        ax.set_axis_off()
        ax.xaxis._axinfo['juggled'] = (0, 1, 2)
        ax.yaxis._axinfo['juggled'] = (1, 0, 2)
        ax.zaxis._axinfo['juggled'] = (2, 1, 0)


    else:
        ax.set_xlabel("X axis")
        ax.set_ylabel("Y axis")
        ax.set_zlabel("Z axis")
        ax.xaxis.set_ticks([-1, 0, 1])
        ax.yaxis.set_ticks([-1, 0, 1])
        ax.zaxis.set_ticks([-1, 0, 1])
        # ax.xaxis.set_ticklabels([-1, 0, 1])
        # ax.yaxis.set_ticklabels([-1, 0, 1])
        # ax.zaxis.set_ticklabels([-1, 0, 1])

    # Title
    if title is not None:
        ax.set_title(str(title) + "\n")
    matplotlib.pyplot.tight_layout()
    if saveFigPath is not None:
        #ax.figure.set_size_inches(4.8 / 2.54, 4.8 / 2.54)
        matplotlib.pyplot.savefig(saveFigPath,bbox_inches='tight',pad_inches=0,dpi=240)
    else:
        matplotlib.pyplot.show()
    plt.clf()
    return maxVal


def computeAngle_pair(args):
    i, data, icoVectors, grains_pair = args
    # Get the orientation vector
    orientationVect = data[i]
    grain_pair = grains_pair[i]
    # Exchange Z and X position - for plotting
    orientationVect = [orientationVect[2], orientationVect[1], orientationVect[0]]
    # Create the result array
    angle = []
    for i in range(len(icoVectors)):
        # Compute the angle between them
        angle.append(numpy.arccos(numpy.clip(numpy.dot(orientationVect, icoVectors[i]), -1, 1)))
    # Get the index
    minIndex = numpy.argmin(angle)
    return minIndex,grain_pair



def pop_particles_that_make_normal_flat(orientations, subDiv=3, reflection=True, maxVal=None, verbose=True, color=None, viewAnglesDeg=[25, 45], title=None, saveFigPath=None, grains_pair=None,past=True,labelled=None):
    """
    这里不是输入orientations[:,2:5],而是输入orientations本体
    Parameters
    ----------
    orientations
    subDiv
    reflection
    maxVal
    verbose
    color
    viewAnglesDeg
    title
    saveFigPath

    Returns
    -------

    """

    import spam.orientations
    matplotlib.rcParams['font.sans-serif'] = 'Arial'
    matplotlib.rcParams['font.size'] = 14

    # Internal function for binning data into the icosphere faces

    def binIcosphere(data, icoVectors, verbose, grains_pair):
        # Create counts array
        counts = numpy.zeros(len(icoVectors))

        index_dict = {}
        for i in range(len(icoVectors)):
            index_dict[i] = []

        # Create progressbar
        if verbose:
            widgets = [progressbar.FormatLabel(""), " ", progressbar.Bar(), " ", progressbar.AdaptiveETA()]
            pbar = progressbar.ProgressBar(widgets=widgets, maxval=len(data))
            pbar.start()
        finishedOrientations = 0

        # Run multiprocessing
        # TODO:开销很大，data和icoVectors应该以某种共用方式定义，使用globalVar或multiprocessing提供的shared_list操作

        returns = run_multi_process(computeAngle_pair,[(i,data,icoVectors,grains_pair) for i in range(len(data))])
        for singlereturn in returns:
            finishedOrientations += 1
            if verbose:
                widgets[0] = progressbar.FormatLabel("{}/{} ".format(finishedOrientations, len(data)))
                pbar.update(finishedOrientations)
            index,pair = singlereturn[0],singlereturn[1]
            counts[index] += 1
            index_dict[index].append(pair)

        return counts,index_dict


    # Get number of points
    orientations.shape[0]
    # Check that they are 3D vectors
    if orientations.shape[1] != 3:
        print("\nspam.helpers.orientationPlotter.plotSphericalHistogram: The input vectors are not 3D")
        return
    # from http://stackoverflow.com/questions/2850743/numpy-how-to-quickly-normalize-many-vectors
    norms = numpy.apply_along_axis(numpy.linalg.norm, 1, orientations)
    orientations = orientations / norms.reshape(-1, 1)
    # Check if we can reflect the vectors
    if reflection:
        orientations = numpy.vstack([orientations, -1 * orientations])
        grains_pair = numpy.vstack([grains_pair, grains_pair])
    # Create the icosphere
    if verbose:
        print("\nspam.helpers.orientationPlotter.plotSphericalHistogram: Creating the icosphere")
    icoVerts, icoFaces, icoVectors = spam.orientations.generateIcosphere(subDiv)

    name = saveFigPath + "-dict.npz"
    if os.path.exists(name):
        contact_datas = np.load(name,allow_pickle=True)
        counts,index_dict = contact_datas['counts'],contact_datas['index_dict']
        index_dict = index_dict.item()
    else:
        # Bin the data
        if verbose:
            print("\nspam.helpers.orientationPlotter.plotSphericalHistogram: Binning the data")
        counts, index_dict = binIcosphere(orientations, icoVectors, verbose=verbose, grains_pair=grains_pair)
        np.savez(name, counts=counts, index_dict=index_dict)



    # Now we are ready to plot
    if verbose:
        print("\nspam.helpers.orientationPlotter.plotSphericalHistogram: Plotting")

    """
    print("接下来计算主集簇")
    top_indices = np.argsort(counts)[-4:-2]
    top_vectors = []

    def fill_along_vector_direction(labelled, vect, n):
        start_point = np.array([235, 235, 235])
        for i in range(min(labelled.shape)):
            point = start_point + i * vect
            if all(0 <= p < s for p, s in zip(point, labelled.shape)):
                labelled[int(point[0]),int(point[1]),int(point[2])] = n
            else:
                print("aaa",i)
                break
        return labelled



    for indice in top_indices:
        #top_vectors.append(icoVectors[indice])

        target_dict = index_dict[indice]

        unique_ids, id_counts = np.unique(target_dict, return_counts=True)
        print(f"共{counts[indice]}个数值落在该区间，涉及{len(unique_ids)}个颗粒，最多出现{max(id_counts)}次")


        import tifffile
        mask = np.isin(labelled, unique_ids)
        labelled[~mask] = 0

        labelled = spam.label.makeLabelsSequential(labelled)

        target_vec = icoVectors[indice]

        target_vec = np.array([target_vec[2],target_vec[1],target_vec[0]])

        filling_num = labelled.max() + 1

        labelled = fill_along_vector_direction(labelled,target_vec,filling_num)

        tifffile.imsave(saveFigPath + "-取向预览图.tif", labelled)

        counts[indice] = 0


        #return
    """



    def put_angle(vectors):
        angles = []
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                v1 = np.array(vectors[i])
                v2 = np.array(vectors[j])
                angle = np.arccos(np.clip(np.dot(v1, v2), -1, 1))
                angle = np.degrees(angle)
                print(angle, end='\t')


    #put_angle(top_vectors)




    if past:
        # Create the figure
        fig = matplotlib.pyplot.figure()
        ax = fig.gca(projection='3d')
        if color is None:
            cmap = matplotlib.pyplot.cm.viridis_r
        else:
            cmap = color
        norm = matplotlib.pyplot.Normalize(vmin=0, vmax=1)
        if maxVal is None:
            maxVal = numpy.max(counts)

        # Don't do it like this, make empty arrays!
        # points = []
        # connectivityMatrix = []

        # Loop through each of the faces
        for i in range(len(icoFaces)):
            # Get the corresponding radius
            radii = counts[i] / maxVal
            if radii != 0:
                # Get the face
                face = icoFaces[i]
                # Get the vertices
                P1 = numpy.asarray(icoVerts[face[0]])
                P2 = numpy.asarray(icoVerts[face[1]])
                P3 = numpy.asarray(icoVerts[face[2]])
                # Extend the vertices as needed by the radius
                P1 = radii * P1 / numpy.linalg.norm(P1)
                P2 = radii * P2 / numpy.linalg.norm(P2)
                P3 = radii * P3 / numpy.linalg.norm(P3)
                # Combine the vertices
                vertices = numpy.asarray([numpy.array([0, 0, 0]), P1, P2, P3])

                # for vertex in vertices:
                # points.append(vertex)
                # connectivityMatrix.append([len(points)-1, len(points)-2, len(points)-3, len(points)-4])

                # Add the points to the scatter3D
                ax.scatter3D(vertices[:, 0], vertices[:, 1], vertices[:, 2], s=0)
                # Create each face
                face1 = numpy.array([numpy.array(vertices[0]), numpy.array(vertices[1]), numpy.array(vertices[2])])
                face2 = [numpy.array(vertices[0]), numpy.array(vertices[1]), numpy.array(vertices[3])]
                face3 = [numpy.array(vertices[0]), numpy.array(vertices[3]), numpy.array(vertices[2])]
                face4 = [numpy.array(vertices[3]), numpy.array(vertices[1]), numpy.array(vertices[2])]

                # Plot each face!
                ax.add_collection3d(
                    Poly3DCollection([face1, face2, face3, face4], facecolors=cmap(norm(radii)), linewidths=0.5,
                                     edgecolors="k"))

        # import spam.helpers
        # spam.helpers.writeUnstructuredVTK(numpy.array(points), numpy.array(connectivityMatrix), cellData={'counts': counts})

        # Extra parameters for the axis
        ax.set_box_aspect([1, 1, 1])
        matplotlib.pyplot.xlim(-1.1, 1.1)
        matplotlib.pyplot.ylim(-1.1, 1.1)
        ax.set_zlim(-1.1, 1.1)
        ax.view_init(25, 45)
        # Set the colorbar
        norm = matplotlib.colors.Normalize(vmin=0, vmax=maxVal)
        sm = matplotlib.pyplot.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        matplotlib.pyplot.colorbar(sm, label="Counts")
        """
        # Remove the ticks labels and lines
        ax = matplotlib.pyplot.gca()
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.zaxis.set_ticklabels([])
        for line in ax.xaxis.get_ticklines():
            line.set_visible(False)
        for line in ax.yaxis.get_ticklines():
            line.set_visible(False)
        for line in ax.zaxis.get_ticklines():
            line.set_visible(False)
        """
        # Title
        if title is not None:
            ax.set_title(str(title) + "\n")
        matplotlib.pyplot.tight_layout()
        if saveFigPath is not None:
            matplotlib.pyplot.savefig(saveFigPath)
        else:
            matplotlib.pyplot.show()


    else:
        # Create the figure
        fig = matplotlib.pyplot.figure()
        # ax = fig.gca(projection='3d')
        ax = fig.add_subplot(projection="3d")

        if color is None:
            cmap = matplotlib.pyplot.cm.RdBu_r
        else:
            cmap = color
        norm = matplotlib.pyplot.Normalize(vmin=0, vmax=1)
        if maxVal is None:
            maxVal = numpy.max(counts)

        # Don't do it like this, make empty arrays!
        # points = []
        # connectivityMatrix = []

        # Loop through each of the faces
        for i in range(len(icoFaces)):
            # Get the corresponding radius
            radii = counts[i] / maxVal
            if radii != 0:
                # Get the face
                face = icoFaces[i]
                # Get the vertices
                P1 = numpy.asarray(icoVerts[face[0]])
                P2 = numpy.asarray(icoVerts[face[1]])
                P3 = numpy.asarray(icoVerts[face[2]])
                # Extend the vertices as needed by the radius
                P1 = radii * P1 / numpy.linalg.norm(P1)
                P2 = radii * P2 / numpy.linalg.norm(P2)
                P3 = radii * P3 / numpy.linalg.norm(P3)
                # Combine the vertices
                vertices = numpy.asarray([numpy.array([0, 0, 0]), P1, P2, P3])

                # for vertex in vertices:
                # points.append(vertex)
                # connectivityMatrix.append([len(points)-1, len(points)-2, len(points)-3, len(points)-4])

                # Add the points to the scatter3D
                ax.scatter3D(vertices[:, 0], vertices[:, 1], vertices[:, 2], s=0)
                # Create each face
                face1 = numpy.array([vertices[0], vertices[1], vertices[2]])
                face2 = numpy.array([vertices[0], vertices[1], vertices[3]])
                face3 = numpy.array([vertices[0], vertices[3], vertices[2]])
                face4 = numpy.array([vertices[3], vertices[1], vertices[2]])

                # Plot each face!
                ax.add_collection3d(Poly3DCollection([face1, face2, face3, face4], facecolors=cmap(norm(radii)), linewidths=0.5, edgecolors="k"))

        # import spam.helpers
        # spam.helpers.writeUnstructuredVTK(numpy.array(points), numpy.array(connectivityMatrix), cellData={'counts': counts})

        # Extra parameters for the axis
        ax.set_box_aspect([1, 1, 1])
        matplotlib.pyplot.xlim(-1.1, 1.1)
        matplotlib.pyplot.ylim(-1.1, 1.1)
        ax.set_zlim(-1.1, 1.1)
        ax.view_init(viewAnglesDeg[0], viewAnglesDeg[1])
        # Set the colorbar
        """
        norm = matplotlib.colors.Normalize(vmin=0, vmax=maxVal)
        sm = matplotlib.pyplot.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        matplotlib.pyplot.colorbar(sm,shrink=1.1)
        """
        hideAxes = False
        if hideAxes:
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            ax.zaxis.set_ticklabels([])

        else:
            ax.set_xlabel("X axis")
            ax.set_ylabel("Y axis")
            ax.set_zlabel("Z axis")
            ax.xaxis.set_ticks([-1, 0, 1])
            ax.yaxis.set_ticks([-1, 0, 1])
            ax.zaxis.set_ticks([-1, 0, 1])
            # ax.xaxis.set_ticklabels([-1, 0, 1])
            # ax.yaxis.set_ticklabels([-1, 0, 1])
            # ax.zaxis.set_ticklabels([-1, 0, 1])

        # Remove the ticks labels and lines
        ax = matplotlib.pyplot.gca()
        for line in ax.xaxis.get_ticklines():
            line.set_visible(False)
        for line in ax.yaxis.get_ticklines():
            line.set_visible(False)
        for line in ax.zaxis.get_ticklines():
            line.set_visible(False)



        # 去掉网格线
        #ax.grid(False)

        # 去掉坐标轴
        #ax.set_axis_off()
        #ax.xaxis._axinfo['juggled'] = (0, 1, 2)
        #ax.yaxis._axinfo['juggled'] = (1, 0, 2)
        #ax.zaxis._axinfo['juggled'] = (2, 1, 0)

        # Title
        if title is not None:
            ax.set_title(str(title) + "\n")
        matplotlib.pyplot.tight_layout()
        if saveFigPath is not None:
            #ax.figure.set_size_inches(4.8 / 2.54, 4.8 / 2.54)
            matplotlib.pyplot.savefig(saveFigPath,bbox_inches='tight',pad_inches=0,dpi=240)
        else:
            matplotlib.pyplot.show()
        plt.clf()
        return maxVal



def plotSphericalHistogram_past(orientations, subDiv=3, reflection=True, maxVal=None, verbose=True, color=None, title=None, saveFigPath=None):
    import spam.orientations
    matplotlib.rcParams['font.sans-serif'] = 'Arial'
    matplotlib.rcParams['font.size'] = 14

    # Internal function for binning data into the icosphere faces

    def binIcosphere(data, icoVectors, verbose):
        # Create counts array
        counts = numpy.zeros(len(icoVectors))

        # Create progressbar
        if verbose:
            widgets = [progressbar.FormatLabel(""), " ", progressbar.Bar(), " ", progressbar.AdaptiveETA()]
            pbar = progressbar.ProgressBar(widgets=widgets, maxval=len(data))
            pbar.start()
        finishedOrientations = 0

        # Run multiprocessing
        # TODO:开销很大，data和icoVectors应该以某种共用方式定义，使用globalVar或multiprocessing提供的shared_list操作
        returns = run_multi_process(computeAngle,[(i,data,icoVectors) for i in range(len(data))])
        for singlereturn in returns:
            finishedOrientations += 1
            if verbose:
                widgets[0] = progressbar.FormatLabel("{}/{} ".format(finishedOrientations, len(data)))
                pbar.update(finishedOrientations)
            index = singlereturn
            counts[index] += 1

        return counts

    # Get number of points
    numberOfPoints = orientations.shape[0]
    # Check that they are 3D vectors
    if orientations.shape[1] != 3:
        print('\nspam.helpers.orientationPlotter.plotSphericalHistogram: The input vectors are not 3D')
        return
        # from http://stackoverflow.com/questions/2850743/numpy-how-to-quickly-normalize-many-vectors
    norms = numpy.apply_along_axis(numpy.linalg.norm, 1, orientations)
    orientations = orientations / norms.reshape(-1, 1)
    # Check if we can reflect the vectors
    if reflection:
        orientations = numpy.vstack([orientations, -1 * orientations])
    # Create the icosphere
    if verbose:
        print('\nspam.helpers.orientationPlotter.plotSphericalHistogram: Creating the icosphere')
    icoVerts, icoFaces, icoVectors = spam.orientations.generateIcosphere(subDiv)
    # Bin the data
    if verbose:
        print('\nspam.helpers.orientationPlotter.plotSphericalHistogram: Binning the data')
    counts = binIcosphere(orientations, icoVectors, verbose=verbose)
    # Now we are ready to plot
    if verbose:
        print('\nspam.helpers.orientationPlotter.plotSphericalHistogram: Plotting')

    # Create the figure
    fig = matplotlib.pyplot.figure()
    ax = fig.gca(projection='3d')
    if color is None:
        cmap = matplotlib.pyplot.cm.viridis_r
    else:
        cmap = color
    norm = matplotlib.pyplot.Normalize(vmin=0, vmax=1)
    if maxVal is None:
        maxVal = numpy.max(counts)

    # Don't do it like this, make empty arrays!
    # points = []
    # connectivityMatrix = []

    # Loop through each of the faces
    for i in range(len(icoFaces)):
        # Get the corresponding radius
        radii = counts[i] / maxVal
        if radii != 0:
            # Get the face
            face = icoFaces[i]
            # Get the vertices
            P1 = numpy.asarray(icoVerts[face[0]])
            P2 = numpy.asarray(icoVerts[face[1]])
            P3 = numpy.asarray(icoVerts[face[2]])
            # Extend the vertices as needed by the radius
            P1 = radii * P1 / numpy.linalg.norm(P1)
            P2 = radii * P2 / numpy.linalg.norm(P2)
            P3 = radii * P3 / numpy.linalg.norm(P3)
            # Combine the vertices
            vertices = numpy.asarray([numpy.array([0, 0, 0]), P1, P2, P3])

            # for vertex in vertices:
            # points.append(vertex)
            # connectivityMatrix.append([len(points)-1, len(points)-2, len(points)-3, len(points)-4])

            # Add the points to the scatter3D
            ax.scatter3D(vertices[:, 0], vertices[:, 1], vertices[:, 2], s=0)
            # Create each face
            face1 = numpy.array([numpy.array(vertices[0]), numpy.array(vertices[1]), numpy.array(vertices[2])])
            face2 = [numpy.array(vertices[0]), numpy.array(vertices[1]), numpy.array(vertices[3])]
            face3 = [numpy.array(vertices[0]), numpy.array(vertices[3]), numpy.array(vertices[2])]
            face4 = [numpy.array(vertices[3]), numpy.array(vertices[1]), numpy.array(vertices[2])]

            # Plot each face!
            ax.add_collection3d(
                Poly3DCollection([face1, face2, face3, face4], facecolors=cmap(norm(radii)), linewidths=0.5,
                                 edgecolors="k"))

    # import spam.helpers
    # spam.helpers.writeUnstructuredVTK(numpy.array(points), numpy.array(connectivityMatrix), cellData={'counts': counts})

    # Extra parameters for the axis
    ax.set_box_aspect([1, 1, 1])
    matplotlib.pyplot.xlim(-1.1, 1.1)
    matplotlib.pyplot.ylim(-1.1, 1.1)
    ax.set_zlim(-1.1, 1.1)
    ax.view_init(25, 45)
    # Set the colorbar
    norm = matplotlib.colors.Normalize(vmin=0, vmax=maxVal)
    sm = matplotlib.pyplot.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    matplotlib.pyplot.colorbar(sm, label="Counts")
    # Remove the ticks labels and lines
    ax = matplotlib.pyplot.gca()
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])
    for line in ax.xaxis.get_ticklines():
        line.set_visible(False)
    for line in ax.yaxis.get_ticklines():
        line.set_visible(False)
    for line in ax.zaxis.get_ticklines():
        line.set_visible(False)
    # Title
    if title is not None:
        ax.set_title(str(title) + "\n")
    matplotlib.pyplot.tight_layout()
    if saveFigPath is not None:
        matplotlib.pyplot.savefig(saveFigPath)
    else:
        matplotlib.pyplot.show()





def contactingLabels_modiefied(lab, labelsList=None, areas=False, boundingBoxes=None, centresOfMass=None):
    maxs = []
    single = False

    if boundingBoxes is None:
        boundingBoxes = spam.label.boundingBoxes(lab)
    if centresOfMass is None:
        centresOfMass = spam.label.centresOfMass(lab, boundingBoxes=boundingBoxes)

    if labelsList is None:
        labelsList = list(range(0, lab.max() + 1))

    # I guess there's only one label
    if type(labelsList) != list:
        labelsList = [labelsList]
        single = True

    contactingLabels = []
    contactingAreas = []

    for label in labelsList:
        # catch zero and return it in order to keep arrays aligned
        if label == 0:
            contactingLabels.append([0])
            if areas:
                contactingAreas.append([0])
        else:
            p1 = spam.label.getLabel(lab, label,
                                     boundingBoxes=boundingBoxes,
                                     centresOfMass=centresOfMass,
                                     margin=2)
            p2 = spam.label.getLabel(lab, label,
                                     boundingBoxes=boundingBoxes,
                                     centresOfMass=centresOfMass,
                                     margin=2,
                                     labelDilate=1)

            dilOnly = numpy.logical_xor(p2['subvol'], p1['subvol'])

            labSlice = lab[p1['slice'][0].start:p1['slice'][0].stop,
                       p1['slice'][1].start:p1['slice'][1].stop,
                       p1['slice'][2].start:p1['slice'][2].stop]

            if (dilOnly.shape == labSlice.shape):
                intersection = dilOnly * labSlice

                counts = numpy.unique(intersection, return_counts=True)

                # Print counts starting from the 1th column since we'll always have a ton of zeros
                # print "\tLabels:\n\t",counts[0][1:]
                # print "\tCounts:\n\t",counts[1][1:]

                contactingLabels.append(counts[0][1:])
                if areas:
                    contactingAreas.append(counts[1][1:])

            else:
                # The particle is near the edge, pad it

                padArray = numpy.zeros((3, 2)).astype(int)
                sliceArray = numpy.zeros((3, 2)).astype(int)
                for i, sl in enumerate(p1['slice']):
                    if sl.start < 0:
                        padArray[i, 0] = -1 * sl.start
                        sliceArray[i, 0] = 0
                    else:
                        sliceArray[i, 0] = sl.start
                    if sl.stop > lab.shape[0]:
                        padArray[i, 1] = sl.stop - lab.shape[0]
                        sliceArray[i, 1] = lab.shape[0]
                    else:
                        sliceArray[i, 1] = sl.stop
                labSlice = lab[sliceArray[0, 0]:sliceArray[0, 1],
                           sliceArray[1, 0]:sliceArray[1, 1],
                           sliceArray[2, 0]:sliceArray[2, 1]]
                labSlicePad = numpy.pad(labSlice, padArray)

                """pad again, bit of extraggted"""
                padding_lengths = [x - y for x, y in zip(dilOnly.shape, labSlicePad.shape)]
                if max(padding_lengths) <= 0:
                    labSlicePadPad = labSlicePad
                else:
                    labSlicePadPad = numpy.pad(labSlicePad, padding_lengths)

                # intersection = dilOnly * labSlicePad

                """crop"""
                a, b, c = dilOnly.shape
                labSlicePadPad = labSlicePadPad[:a, :b, :c]
                """edit↑"""
                intersection = dilOnly * labSlicePadPad

                counts = numpy.unique(intersection, return_counts=True)
                contactingLabels.append(counts[0][1:])
                if areas:
                    contactingAreas.append(counts[1][1:])

    # Flatten if it's a list with only one object
    if single:
        contactingLabels = contactingLabels[0]
        if areas:
            contactingAreas = contactingAreas[0]
    # Now return things
    if areas:
        return contactingLabels, contactingAreas
    else:
        return contactingLabels
def detectOverSegmentation_modiefied(lab):
    # Get the labels
    labels = list(range(0, lab.max() + 1))
    # Compute the volumes
    vol = spam.label.volumes(lab)
    # Compute the eq diameter
    eqDiam = spam.label.equivalentRadii(lab)
    # Compute the areas
    contactLabels = contactingLabels_modiefied(lab, areas=True)
    # Create result list
    overSegCoeff = []
    sharedLabel = []
    for label in labels:
        if label == 0:
            overSegCoeff.append(0)
            sharedLabel.append(0)
        else:
            # Check if there are contacting areas and volumes
            if len(contactLabels[1][label]) > 0 and vol[label] > 0:
                # We have areas on the list, compute the area
                maxArea = numpy.max(contactLabels[1][label])
                # Get the label for the max contacting area
                maxLabel = contactLabels[0][label][numpy.argmax(contactLabels[1][label])]
                # Compute the coefficient
                overSegCoeff.append(maxArea * eqDiam[label] / vol[label])
                # Add the label
                sharedLabel.append(maxLabel)
            else:
                overSegCoeff.append(0)
                sharedLabel.append(0)
    overSegCoeff = numpy.array(overSegCoeff)
    sharedLabel = numpy.array(sharedLabel)
    return overSegCoeff, sharedLabel

def computeConvexVolume(args):
        lab,label,boundingBoxes,centresOfMass = args
        labelI = spam.label.getLabel(lab, label, boundingBoxes=boundingBoxes, centresOfMass=centresOfMass)
        subvol = labelI['subvol']
        points = numpy.transpose(numpy.where(subvol))
        try:
            hull = scipy.spatial.ConvexHull(points)
            deln = scipy.spatial.Delaunay(points[hull.vertices])
            idx = numpy.stack(numpy.indices(subvol.shape), axis=-1)
            out_idx = numpy.nonzero(deln.find_simplex(idx) + 1)
            hullIm = numpy.zeros(subvol.shape)
            hullIm[out_idx] = 1
            hullVol = spam.label.volumes(hullIm)
            return label, hullVol[-1]
        except:
            return label, 0

def convexVolume_modiefied(lab, boundingBoxes=None, centresOfMass=None, volumes=None, nProcesses=nProcessesDefault, verbose=True):

    lab = lab.astype(labelType)

    # Compute boundingBoxes if needed
    if boundingBoxes is None:
        boundingBoxes = spam.label.boundingBoxes(lab)
    # Compute centresOfMass if needed
    if centresOfMass is None:
        centresOfMass = spam.label.centresOfMass(lab)
    # Compute volumes if needed
    if volumes is None:
        volumes = spam.label.volumes(lab)
    # Compute number of labels
    nLabels = lab.max()

    # Result array
    convexVolume = numpy.zeros(nLabels+1, dtype='float')

    if verbose:
        widgets = [progressbar.FormatLabel(''), ' ', progressbar.Bar(), ' ', progressbar.AdaptiveETA()]
        pbar = progressbar.ProgressBar(widgets=widgets, maxval=nLabels)
        pbar.start()
        finishedNodes = 0

    results = run_multi_process(computeConvexVolume,[(lab,i,boundingBoxes,centresOfMass) for i in range(1, nLabels + 1)])
    for returns in results:
        if verbose:
            finishedNodes += 1
            pbar.update(finishedNodes)
        convexVolume[returns[0]] = returns[1]
    if verbose:
        pbar.finish()
    return convexVolume
def detectUnderSegmentation_modiefied(lab, nProcesses=nProcessesDefault, verbose=True):
    # Compute the volume
    vol = spam.label.volumes(lab)
    print(nProcesses)
    convexVol = convexVolume_modiefied(lab, verbose=verbose, nProcesses=nProcesses)
    # Set the volume of the void to 0 to avoid the division by zero error
    vol[0] = 1
    # Compute the underSegmentation Coefficient
    underSegCoeff = convexVol / vol
    # Set the coefficient of the void to 0
    underSegCoeff[0] = 0
    return underSegCoeff


def fixUndersegmentation_modiefied(lab, imGrey, targetLabels, underSegCoeff, boundingBoxes=None, centresOfMass=None, imShowProgress=False, verbose=True, disableCoeffCheck=False):
    """

    Parameters
    ----------
    lab
    imGrey
    targetLabels
    underSegCoeff
    boundingBoxes
    centresOfMass
    imShowProgress
    verbose
    disableCoeffCheck:确定传入标签一定是欠分割，不检查重划分后是否欠分割系数降低

    Returns
    -------

    """
    # Usual checks
    if boundingBoxes is None:
        boundingBoxes = spam.label.boundingBoxes(lab)
    if centresOfMass is None:
        centresOfMass = spam.label.centresOfMass(lab)
    # Check if imGrey is normalised (limits [0,1])
    print(imGrey.max())
    print(imGrey.min())
    if imGrey.max() > 1 or imGrey.min() < 0:
        print('\n spam.label.fixUndersegmentation(): imGrey is not normalised. Limits exceed [0,1]')
        return
    # Start counters
    labelCounter = numpy.max(lab)
    labelDummy = numpy.zeros(lab.shape)
    successCounter = 0
    finishedLabels = 0
    if verbose:
        widgets = [progressbar.FormatLabel(''), ' ', progressbar.Bar(), ' ', progressbar.AdaptiveETA()]
        pbar = progressbar.ProgressBar(widgets=widgets, maxval=len(targetLabels))
        pbar.start()
    # Main loop
    for label in targetLabels:
        # Get the subset
        labelData = spam.label.getLabel(lab, label, margin = 1, boundingBoxes=boundingBoxes, centresOfMass=centresOfMass, extractCube=True)
        # Get the slice on the greyscale
        imGreySlice = imGrey[labelData['sliceCube'][0].start:labelData['sliceCube'][0].stop,
                             labelData['sliceCube'][1].start:labelData['sliceCube'][1].stop,
                             labelData['sliceCube'][2].start:labelData['sliceCube'][2].stop]
        # Mask the imGreySubset
        #更改
        try:
            greySubset = imGreySlice*labelData['subvol']
        except:
            raise Exception("hardmard error")
        # Create seeds
        #2021-08-02 GP: Maybe this can be changed by just a serie of binary erosion?
        seeds = spam.label.watershed(greySubset >= 0.75)
        # Do we have seeds?
        if numpy.max(seeds) < 1:
            # The threshold was too harsh on the greySubset and there are no seeds
            # We shouldn't change this label
            passBool = 'Decline'
        else:
            # We have at least one seed, Run watershed again with markers
            imLabSubset = spam.label.watershed(labelData['subvol'], markers = seeds)
            if not disableCoeffCheck:
                # Run again the underSegCoeff for the subset
                #更改
                res = detectUnderSegmentation_modiefied(imLabSubset, verbose=False)
                # Safety check - do we have any labels at all?
                if len(res) > 2:
                    # We have at least one label
                    # Check if it should pass or not - is the new underSegCoeff of all the new labels less than the original coefficient?
                    if all(map(lambda x: x < underSegCoeff[label], res[1:])):
                        # We can modify this label
                        passBool = 'Accept'
                        successCounter += 1
                        # Remove the label from the original label image
                        lab = spam.label.removeLabels(lab, [label])
                        # Assign the new labels to the grains
                        # Create a subset to fill with the new labels
                        imLabSubsetNew = numpy.zeros(imLabSubset.shape)
                        for newLab in numpy.unique(imLabSubset[imLabSubset != 0]):
                            imLabSubsetNew = numpy.where(imLabSubset==newLab, labelCounter + 1, imLabSubsetNew)
                            labelCounter += 1
                        # Create a disposable dummy sample to allocate the grains
                        labelDummyUnit = numpy.zeros(lab.shape)
                        #Alocate the grains
                        labelDummyUnit[labelData['sliceCube'][0].start:labelData['sliceCube'][0].stop,
                                       labelData['sliceCube'][1].start:labelData['sliceCube'][1].stop,
                                       labelData['sliceCube'][2].start:labelData['sliceCube'][2].stop] = imLabSubsetNew
                        # Add the grains
                        labelDummy = labelDummy + labelDummyUnit
                    else:
                        # We shouldn't change this label
                        passBool = 'Decline'
                else:
                    # We shouldn't change this label
                    passBool = 'Decline'
            else:
                # We can modify this label
                passBool = 'Accept'
                successCounter += 1
                # Remove the label from the original label image
                lab = spam.label.removeLabels(lab, [label])
                # Assign the new labels to the grains
                # Create a subset to fill with the new labels
                imLabSubsetNew = numpy.zeros(imLabSubset.shape)
                for newLab in numpy.unique(imLabSubset[imLabSubset != 0]):
                    imLabSubsetNew = numpy.where(imLabSubset == newLab, labelCounter + 1, imLabSubsetNew)
                    labelCounter += 1
                # Create a disposable dummy sample to allocate the grains
                labelDummyUnit = numpy.zeros(lab.shape)
                # Alocate the grains
                labelDummyUnit[labelData['sliceCube'][0].start:labelData['sliceCube'][0].stop,
                labelData['sliceCube'][1].start:labelData['sliceCube'][1].stop,
                labelData['sliceCube'][2].start:labelData['sliceCube'][2].stop] = imLabSubsetNew
                # Add the grains
                labelDummy = labelDummy + labelDummyUnit

        if imShowProgress:
            # Enter graphical mode
            # Change the labels to show different colourss
            fig=plt.figure()
            # Plot
            plt.subplot(3,2,1)
            plt.gca().set_title('Before')
            plt.imshow( labelData['subvol'][labelData['subvol'].shape[0]//2, :, :], cmap="Greys_r" )
            plt.subplot(3,2,2)
            plt.gca().set_title('After')
            plt.imshow( imLabSubset[imLabSubset.shape[0]//2, :, :], cmap="cubehelix" )
            plt.subplot(3,2,3)
            plt.imshow( labelData['subvol'][:, labelData['subvol'].shape[1]//2,:], cmap="Greys_r" )
            plt.subplot(3,2,4)
            plt.imshow( imLabSubset[:, imLabSubset.shape[1]//2, :], cmap="cubehelix" )
            plt.subplot(3,2,5)
            plt.imshow( labelData['subvol'][:, :, labelData['subvol'].shape[2]//2], cmap="Greys_r" )
            plt.subplot(3,2,6)
            plt.imshow( imLabSubset[:, :, imLabSubset.shape[2]//2], cmap="cubehelix" )
            fig.suptitle(r'Label {}. Status: $\bf{}$'.format(label, passBool), fontsize='xx-large')
            plt.show()
        if verbose:
            finishedLabels += 1
            pbar.update(finishedLabels)
    # We finish, lets add the new grains to the labelled image
    lab = lab + labelDummy
    # Update the labels
    lab = spam.label.makeLabelsSequential(lab)
    if verbose:
        pbar.finish()
        print('\n spam.label.fixUndersegmentation(): From {} target labels, {} were modified'.format( len(targetLabels), successCounter))
    return lab