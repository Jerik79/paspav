# PaSpaV

**Pa**rameter **Spa**ce **V**isualiser for polygonal curves, based on [Matplotlib](https://matplotlib.org/).


## Features

- PaSpaV visualises the parameter space terrain of two polygonal curves as a contour plot.
  The height values in such a terrain correspond to distances between curve points.

- PaSpaV allows selecting the norm that is used for height calculations.
  It supports the $1$-norm, the $2$-norm and the $\infty$-norm in every dimension.
  In 2D it makes polygonal norms available too.

- PaSpaV can generate random polygonal curves and load curves from CSV files.
  It can also save curves as CSV files and save contour plots as image files in all formats supported by Matplotlib.


## Usage

First, install the dependencies listed in [environment.yml](./environment.yml).
Then you should be able to use the PaSpaV CLI.

```console
$ python paspav.py --help
```

The above command shows all general options, many of which are also configurable interactively from inside the PaSpaV GUI.
That GUI can be invoked with randomly generated polygonal curves.

```console
$ python paspav.py random
```

Inside the PaSpaV GUI, which visualises the parameter space of the curves, you can repeatedly generate more random curves and save them as CSV files.
Those files can later be loaded into the GUI again.

```console
$ python paspav.py load curve1.csv curve2.csv
```

Finally, you can use the PaSpaV CLI to save the parameter space of loaded curves as an image file.

```console
$ python paspav.py load curve1.csv curve2.csv saveimg image.pdf
```

The above subcommands accept the ```--help``` option as well, giving an overview of all their options.
