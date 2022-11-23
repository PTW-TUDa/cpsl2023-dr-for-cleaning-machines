#!/usr/bin/env python
""" The script can be used to read data from different servers in the ETA-Factory.
It can write output to CSV files and/or publish to a different OPC UA server.
"""

import argparse
import asyncio
import os
import pathlib

from eta_utility import get_logger, SelfsignedKeyCertPair, PEMKeyCertPair
from eta_utility.connectors import Node, connections_from_nodes, sub_handlers
import keyboard

log = get_logger(level=2, format="logname")


def main():
    args = parse_args()

    execution_loop(
        args.nodes_file, args.nodes_sheet, args.output_file, args.publish_opcua, args.stop_after,
        args.sub_interval, args.write_interval, args.eneffco_usr, args.eneffco_pw, args.key_path,
        args.key_passphrase, args.cert_path, args.verbosity
    )


def parse_args():
    """ Parse command line arguments (see help for description).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "nodes_file", action="store", type=str,
        help="Excel file to read nodes from."
    )
    parser.add_argument(
        "nodes_sheet", action="store", type=str,
        help="Name of the Excel sheet specifying the nodes."
    )
    parser.add_argument(
        "--output_file", action="store", type=str, default=None,
        help = "Path to the CSV output file.")
    parser.add_argument("--publish_opcua", action = "store", type = bool, default = False,
                        help = "Set up an OPC UA server to re-publish recorded values. NOT IMPLEMENTED YET.")
    parser.add_argument(
        "--stop_after", action="store", type=int, default=None,
        help="Stop recording after X seconds."
    )
    parser.add_argument(
        "--sub_interval", action="store", type=float, default=1,
        help="Subscript read interval (or polling interval) in seconds."
    )
    parser.add_argument(
        "--write_interval", action="store", type=float, default=1,
        help="Writing interval in seconds for writing to CSV file."
    )
    parser.add_argument(
        "--eneffco_usr", action="store", type=str, default=None,
        help="EnEffCo user name."
    )
    parser.add_argument(
        "--eneffco_pw", action="store", type=str, default=None,
        help="EnEffCo password."
    )
    parser.add_argument(
        "--key_path", action="store", type=str, default=None,
        help="Path to a PEM format RSA key file. If this is not provided, a temporary "
        "key will be generated automatically."
    )
    parser.add_argument(
        "--key_passphrase", action="store", type=str, default=None,
        help="Passphrase for the RSA key. If this is not provided, the key must be unencrypted."
    )
    parser.add_argument(
        "--cert_path", action="store", type=str, default=None,
        help="Path to a PEM format x509 certificate file. If this is not provided, a "
        "temporary selfsigned certificate will be generated."
    )
    parser.add_argument(
        "--verbosity", action="store", type=int, default=2,
        help="Verbosity level (between 0 - no output and 4 - debug)."
    )

    return parser.parse_args()


async def logger(interval):
    """ Print info message every interval seconds to show that the program continues to work

    :param int interval: Interval for printing the message in seconds
    :return:
    """
    step = 0
    while True:
        await asyncio.sleep(interval)
        step += interval
        print("Logging data for {} s".format(step))


async def stop_execution(sleep_time):
    """ Stop execution after the specified time interval.

    :param int sleep_time: Time interval in seconds.
    """
    await asyncio.sleep(sleep_time)


async def stop_keyboard(key = "q"):
    """ Stop execution if a key is pressed

    :param str key: Key to be pressed (default: "q")
    :return:
    """
    while True:
        if keyboard.is_pressed(key):
            break
        await asyncio.sleep(0)


def execution_loop(
        nodes_file, nodes_sheet, output_file = None, publish_opcua = False,
        stop_after = None, sub_interval = 1, write_interval = 1, eneffco_usr = None, eneffco_pw = None,
        key_path = None, key_passphrase = None, cert_path = None,
        verbosity = 2
):
    """ Execute the subscription and publishing loop

    :param str nodes_file: Path to excel sheet with node specification
    :param str nodes_sheet: excel sheet name
    :param str output_file: Path to the CSV output file (optional) - One of output_file or publish_opcua is required.
    :param bool publish_opcua: Set true, if data should be published to a local OPC UA Server - One of output_file or
    publish_opcua is required.
    :param int stop_after: Stop recording data automatically after X seconds
    :param float sub_interval: Interval for subscription data.
    :param float write_interval: Interval for writing to CSV file
    :param str eneffco_usr: EnEffCo user name.
    :param str eneffco_pw: EnEffCo password.
    :param str key_path: Path to a PEM format RSA key file.
    :param str key_passphrase: Passphrase for the RSA key file.
    :param str cert_path: Path to a PEM format x509 certificate file.
    :param int verbosity: Verbosity level (between 0 - no output and 4 - debug).
    """
    log.setLevel(verbosity * 10)

    nodes = Node.from_excel(nodes_file, nodes_sheet, fail=False)

    # Initialize a certificate and key pair from file or generate a selfsigned pair.
    if key_path is not None or cert_path is not None:
        if key_path is None or cert_path is None:
            raise ValueError(
        "If specifying one of key_path and cert_path, the other must be specified as well "
        "(only one value found)."
        )
        key_cert = PEMKeyCertPair(key_path, cert_path, key_passphrase)

    else:
        key_cert = SelfsignedKeyCertPair("opc_client")

    with key_cert.tempfiles() as kc:
        connections = connections_from_nodes(nodes, usr=eneffco_usr, pwd=eneffco_pw, key_cert=kc)

        # Start handler
        subscription_handler = sub_handlers.MultiSubHandler()

        if output_file is None and publish_opcua is None:
            raise ValueError("Specify at least one of output_file or publish_opcua")

        if output_file is not None:
            output_file = pathlib.Path(output_file)
            if output_file.is_file() or output_file.is_dir():
                try:
                    os.remove(output_file)
                except FileNotFoundError:
                    pass

            subscription_handler.register(sub_handlers.CsvSubHandler(output_file, write_interval=write_interval))

        if publish_opcua is not None:
            pass  # This is currently missing from utility functions.
        # subscription_handler.register(sub_handlers.OpcUaSubHandler())

        loop = asyncio.get_event_loop()
        loop.create_task(logger(10))

        try:
            for host, connection in connections.items():
                # Start connections without passing on interrupt signals
                try:
                    connection.subscribe(subscription_handler, interval=sub_interval)
                except ConnectionError as e:
                    log.warning(str(e))

            print("Starting processing loop")
            if stop_after is not None:
                print("Process will stop after {} s.".format(stop_after))
                loop.run_until_complete(stop_execution(stop_after))
            else:
                print("Use q to stop recording data (It might take some time to react).")
                loop.run_until_complete(stop_keyboard("q"))
                print("Detected key press, stopping execution.")

        finally:
            print("Closing connections and handlers")
            for host, connection in connections.items():
                connection.close_sub()

            subscription_handler.close()


if __name__ == "__main__":
    main()
