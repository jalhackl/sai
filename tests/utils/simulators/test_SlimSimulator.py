import os
import pytest
import shutil
from sai.utils.multiprocessing import mp_manager
from sai.utils.simulators import SlimSimulator
from sai.utils.generators import RandomNumberGenerator
import tskit



@pytest.fixture
def sim_params():
    return {
        "output_dir": os.path.join("tests", "test_SlimSimulator"),
        "slim_script_folder": os.path.join(
            "examples", "slim", "racimo"
        ),
        "slim_script_name": "racimo_default.slim",
        "nref": 108,
        "ntgt": 99,
        "ref_id": "p1",
        "tgt_id": "p4",
        "src_id": "p2",
        "seq_len": 40000,
        "adm_amount": 0.02,
        "output_prefix": "slim_sim",
        "is_phased": True,
        "scaling_factor": 10,
        "basic_mut_rate": 1.5e-8,
        "neutral_mut_rate": 1.5e-8,
        "archaic_sample_time": "oldest",
        "nsrc": 1,
        "create_tracts_file": False,
        "create_mutation_check_file": True,
        "slim_mutation_nr": 2,
        "extra_mutations": False,
        "resample": 0,
        "chromosome": "1",
        "return_finished_trees": False,
        "simplify_input_trees": True,
        "selection_coefficient": 0.1,
        "recombination_rate": 1e-8,
        "ancestral_Ne": 10000,
        "initial_sweep_frequency": 0,
        "recapitate_slim_sim": True,
        "src_sample_generations": 1500,
    }


@pytest.fixture
def cleanup_output_dir(request, sim_params):
    #pass
    # Setup (nothing to do before the test)
    yield  # Hand over control to the test
    # Teardown
    shutil.rmtree(sim_params["output_dir"], ignore_errors=True)


def compare_files(file1, file2):
    with open(file1, "r") as f1, open(file2, "r") as f2:
        file1_content = f1.read()
        file2_content = f2.read()
        assert file1_content == file2_content, "Files do not match."


def test_SlimSimulator(sim_params, cleanup_output_dir):
    nprocess = 1
    nrep = 4

    os.makedirs(sim_params["output_dir"], exist_ok=True)
    print("folder created")
    print(sim_params["output_dir"])

    simulator = SlimSimulator(**sim_params)
    generator = RandomNumberGenerator(nrep=nrep, seed=12345)

    mp_manager(job=simulator, data_generator=generator, nprocess=nprocess)
    for i in range(nrep):
        ref_ind_file = os.path.join(
            sim_params["output_dir"],
            f"{i}",
            f"{sim_params['output_prefix']}.{i}.ref.ind.list",
        )
        tgt_ind_file = os.path.join(
            sim_params["output_dir"],
            f"{i}",
            f"{sim_params['output_prefix']}.{i}.tgt.ind.list",
        )
        src_ind_file = os.path.join(
            sim_params["output_dir"],
            f"{i}",
            f"{sim_params['output_prefix']}.{i}.src.ind.list",
        )
        ts_file = os.path.join(
            sim_params["output_dir"],
            f"{i}",
            f"{sim_params['output_prefix']}.{i}.ts",
        )
        mut_file = os.path.join(
            sim_params["output_dir"],
            f"{i}",
            f"{sim_params['output_prefix']}.{i}.tgt.mut.list",
        )
        vcf_file = os.path.join(
            sim_params["output_dir"], f"{i}", f"{sim_params['output_prefix']}.{i}.vcf"
        )

        expected_dir = "tests/expected_results/simulators/SlimSimulator"

        expected_ref_ind_file = os.path.join(
            expected_dir, f"{i}", f"{sim_params['output_prefix']}.{i}.ref.ind.list"
        )
        expected_tgt_ind_file = os.path.join(
            expected_dir, f"{i}", f"{sim_params['output_prefix']}.{i}.tgt.ind.list"
        )

        expected_src_ind_file = os.path.join(
            expected_dir, f"{i}", f"{sim_params['output_prefix']}.{i}.src.ind.list"
        )

        expected_ts_file = os.path.join(
            expected_dir, f"{i}", f"{sim_params['output_prefix']}.{i}.ts"
        )
        expected_mut_file = os.path.join(
            expected_dir, f"{i}", f"{sim_params['output_prefix']}.{i}.tgt.mut.list"
        )
        expected_vcf_file = os.path.join(
            expected_dir, f"{i}", f"{sim_params['output_prefix']}.{i}.vcf"
        )
        
        ts_test = tskit.load(ts_file)
        ts_expected = tskit.load(expected_ts_file)

        assert list(ts_test.sites()) == list(ts_expected.sites())

        compare_files(ref_ind_file, expected_ref_ind_file)
        compare_files(tgt_ind_file, expected_tgt_ind_file)
        compare_files(src_ind_file, expected_src_ind_file)
        compare_files(mut_file, expected_mut_file)
        compare_files(vcf_file, expected_vcf_file)
