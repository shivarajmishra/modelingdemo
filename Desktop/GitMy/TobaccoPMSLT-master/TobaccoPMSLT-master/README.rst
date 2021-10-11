
Installation
------------

Open up a terminal and run::
	$> conda create --name=2105_tobacco python=3.6
	$> conda activate 2105_tobacco
	(examplePMSLT) $> git clone https://github.com/population-interventions/TobaccoPMSLT
	(examplePMSLT) $> cd TobaccoPMSLT
	(examplePMSLT) $> pip install -e .


Make the data artifacts
------------
Run::

	(examplePMSLT) $> make_artifacts minimal


Run a single simulation
------------
Run::

	(2105_tobacco) $> simulate run -v model_specs/test_model.yaml

Results are stored in TobaccoPMSLT/results

Run::

	(2105_tobacco) $> run_uncertainty_analysis model_specs/test_model.yaml -d 0

With profiling to profile.

Package versions (via pip freeze)
------------
aiocontextvars==0.2.2
certifi==2020.12.5
click==8.0.0
colorama==0.4.4
contextvars==2.4
decorator==4.4.2
immutables==0.15
Jinja2==3.0.0
loguru==0.5.3
MarkupSafe==2.0.0
networkx==2.5.1
numexpr==2.7.3
numpy==1.19.5
pandas==1.1.5
python-dateutil==2.8.1
pytz==2021.1
PyYAML==5.4.1
scipy==1.5.4
six==1.16.0
tables==3.6.1
vivarium==0.10.4
## !! Could not determine repository location
vivarium-pmslt==1.0.0
win32-setctime==1.0.3
wincertstore==0.2
