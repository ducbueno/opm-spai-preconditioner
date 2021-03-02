#include <fstream>
#include <iostream>
#include <iterator>
#include <boost/program_options.hpp>
#include "openclBackend.hpp"

using namespace std;
namespace po = boost::program_options;

int main(int argc, char* argv[]){
    int verbosity;
    unsigned int platformID = 0;
    unsigned int deviceID = 0;
    double tau;

    po::options_description desc("Allowed options");
    desc.add_options()("verbosity", po::value<int>(), "verbosity");
    desc.add_options()("tau", po::value<double>(), "tau");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    verbosity = vm["verbosity"].as<int>();
    tau = vm["tau"].as<double>();

    openclBackend oclBackend(verbosity, platformID, deviceID, tau);
    oclBackend.run();

    return 0;
}
