# Contributing Guidelines

## Developer Dependencies

To get started, please do the following:

1. Install [python-poetry](https://python-poetry.org/docs/#installation), used for building our distribution wheels
2. Install [pre-commit](https://pre-commit.com/#install), used to run pre-commit hooks like code linting and unit tests
3. Run `$ pre-commit install` in the root of the directory to setup the hook scripts.

Additionally, we strongly recommend using a virtual environment to avoid dependency conflicts.


## Test Suites

### Unit Tests

We define a variety of unit tests in the `./test` directory which match `test*.py` to improve the confidence in our code. These can be run with:

```bash
make test
```

### Integration Tests

We also include an additional test suite which tests that the `l2r` environment is appropriately interfacing with the racing simulator. These tests are run in an isolated Docker network via [docker-compose](https://docs.docker.com/compose/). At a high level, we want to validate that our interfaces are functioning correctly by making actual network calls to the simulator. All files in `./test` that match `sim_test*.py` will be run when running these tests.

Running the integration tests requires:

* Having [Docker](https://docs.docker.com/) and [Docker-compose](https://docs.docker.com/compose/)
* Access to the Arrival simulator Docker image, currently `arrival-sim:0.7.1.188691`

To run integration tests, you must have the Arrival simulator Docker image and a sufficient GPU to run this image. The following script will cleanly build the `l2r` package, create a new `l2r` Docker image, and launch containers which run the integration tests:

```bash
make clean-sim-test
```
