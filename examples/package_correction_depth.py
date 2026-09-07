#!/usr/bin/env python3
"""Build portable correction40 inputs or assemble their authenticated H200 fixture."""
from pathlib import Path
import argparse
import json
from docprune.correction_corpus import assemble, build_package


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    build=sub.add_parser('build')
    build.add_argument('--source',type=Path,required=True)
    build.add_argument('--output',type=Path,required=True)
    build.add_argument('--ledger',type=Path,required=True)
    build.add_argument('--manifest-only',action='store_true')
    assembly=sub.add_parser('assemble')
    assembly.add_argument('--package',type=Path,required=True)
    assembly.add_argument('--output',type=Path,required=True)
    assembly.add_argument('--case-id',action='append',default=[])
    assembly.add_argument('--case-ids-file',type=Path,
                          help='JSON list of explicitly selected case IDs')
    assembly.add_argument('--reuse-root',type=Path,action='append',default=[])
    for name in ['config','run-config','index-manifest']:
        assembly.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if args.command=='build':
        result=build_package(args.source,args.output,args.ledger,include_assets=not args.manifest_only)
        print(json.dumps({'cases':result['question_count'],'assets':len(result['assets']),'output':str(args.output)}))
    else:
        if args.case_ids_file:
            if args.case_id:
                parser.error('use either --case-id or --case-ids-file')
            args.case_id = json.loads(args.case_ids_file.read_text())
            if not isinstance(args.case_id, list) or not args.case_id or any(
                    not isinstance(value, str) for value in args.case_id):
                parser.error('case-ids-file must contain a nonempty JSON string list')
        print(json.dumps(assemble(args.package,args.output,args.reuse_root,config=args.config,run_config=args.run_config,index_manifest=args.index_manifest,case_ids=args.case_id),indent=2))


if __name__=='__main__':
    main()
