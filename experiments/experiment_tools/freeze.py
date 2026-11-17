"""SHA-256 inventories of explicitly selected files and directory trees.

This records input bytes, not a model compatibility/source fingerprint.  Tree
hashes cover sorted POSIX relative paths, directory entries (including empty
directories), file sizes and file SHA-256 values.  The list is encoded as UTF-8
JSON with sorted object keys, no insignificant whitespace, and literal Unicode.
Symlinks and Windows reparse points are rejected instead of followed.
"""
from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path,PurePosixPath
import stat


_KIND='input_sha256_inventory'
_TREE_FORMAT='relative-path-kind-size-sha256-canonical-json-v1'


def _require(condition,message):
    if not condition:raise ValueError(message)


def _tree_hash(members):
    payload=json.dumps(members,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def _plain_stat(path):
    value=path.lstat()
    reparse=getattr(stat,'FILE_ATTRIBUTE_REPARSE_POINT',0)
    _require(not stat.S_ISLNK(value.st_mode) and not (getattr(value,'st_file_attributes',0)&reparse),
             'Cannot freeze symlink or reparse point: '+str(path))
    _require(stat.S_ISREG(value.st_mode) or stat.S_ISDIR(value.st_mode),
             'Only regular files and directories can be frozen: '+str(path))
    return value


def _file(path):
    before=_plain_stat(path)
    _require(stat.S_ISREG(before.st_mode),'Input changed file type while freezing: '+str(path))
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    after=_plain_stat(path)
    fields=('st_dev','st_ino','st_size','st_mtime_ns')
    _require(all(getattr(before,key)==getattr(after,key) for key in fields),
             'Input changed while hashing: '+str(path))
    return dict(kind='file',size_bytes=after.st_size,sha256=digest.hexdigest())


def _entry(path):
    value=_plain_stat(path)
    if stat.S_ISREG(value.st_mode):return dict(path=str(path),**_file(path))
    members=[]
    def walk_error(error):raise error
    for current,dirs,files in os.walk(path,followlinks=False,onerror=walk_error):
        dirs.sort();files.sort()
        for name in dirs:
            child=Path(current)/name
            _require(stat.S_ISDIR(_plain_stat(child).st_mode),'Input changed directory type: '+str(child))
            members.append(dict(path=child.relative_to(path).as_posix(),kind='directory'))
        for name in files:
            child=Path(current)/name
            members.append(dict(path=child.relative_to(path).as_posix(),**_file(child)))
    members.sort(key=lambda item:item['path'])
    return dict(path=str(path),kind='directory',sha256=_tree_hash(members),members=members)


def freeze_inputs(paths):
    """Return a JSON-compatible SHA-256 inventory; paths are stored absolute.

    The caller chooses all inputs.  This does not discover imports/dependencies
    or create an atomic filesystem snapshot.  Do not include a live output tree.
    """
    _require(not isinstance(paths,(str,bytes,os.PathLike)) and hasattr(paths,'__iter__'),
             'paths must be an iterable of paths')
    entries=[];seen=set()
    try:
        for value in paths:
            _require(isinstance(value,(str,os.PathLike)),'Each input must be a path')
            path=Path(value)
            _plain_stat(path)  # Check the supplied leaf before resolve follows links.
            path=path.resolve(strict=True);identity=os.path.normcase(str(path))
            _require(identity not in seen,'Duplicate frozen input: '+str(path))
            seen.add(identity);entries.append(_entry(path))
    except OSError as error:
        raise ValueError('Cannot inventory frozen input: '+str(error)) from error
    entries.sort(key=lambda item:item['path'])
    return dict(schema_version=1,kind=_KIND,hash_algorithm='sha256',tree_hash_format=_TREE_FORMAT,
                entries=entries)


def _validate_snapshot(snapshot):
    """Pure schema validation shared by the dry-run plan validator."""
    _require(isinstance(snapshot,Mapping),'Frozen inputs must be an inventory object')
    _require(set(snapshot)=={'schema_version','kind','hash_algorithm','tree_hash_format','entries'},
             'Unknown or missing frozen-input inventory fields')
    _require(type(snapshot['schema_version']) is int and snapshot['schema_version']==1
             and snapshot['kind']==_KIND and snapshot['hash_algorithm']=='sha256'
             and snapshot['tree_hash_format']==_TREE_FORMAT,'Unsupported input inventory/hash format')
    _require(isinstance(snapshot['entries'],list),'Inventory entries must be a list')
    def digest(value):return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)
    seen=set();paths=[]
    for item in snapshot['entries']:
        _require(isinstance(item,dict),'Inventory entry must be an object')
        path=item.get('path');kind=item.get('kind')
        _require(isinstance(path,str) and path and '\0' not in path and Path(path).is_absolute(),
                 'Frozen input paths must be absolute')
        identity=os.path.normcase(path)
        _require(identity not in seen,'Duplicate inventory path: '+path);seen.add(identity);paths.append(path)
        _require(kind in ('file','directory') and digest(item.get('sha256')),'Invalid inventory entry/hash')
        if kind=='file':
            _require(set(item)=={'path','kind','size_bytes','sha256'} and type(item['size_bytes']) is int
                     and item['size_bytes']>=0,'Invalid frozen file entry')
            continue
        _require(set(item)=={'path','kind','sha256','members'} and isinstance(item['members'],list),
                 'Invalid frozen directory entry')
        names=[]
        for member in item['members']:
            _require(isinstance(member,dict),'Tree member must be an object')
            name=member.get('path');member_kind=member.get('kind')
            _require(isinstance(name,str) and name and '\0' not in name and '\\' not in name
                     and not PurePosixPath(name).is_absolute() and name not in ('.','..')
                     and '..' not in PurePosixPath(name).parts and str(PurePosixPath(name))==name,
                     'Tree members need canonical relative POSIX paths')
            names.append(name)
            if member_kind=='directory':_require(set(member)=={'path','kind'},'Invalid directory tree member')
            else:_require(member_kind=='file' and set(member)=={'path','kind','size_bytes','sha256'}
                          and type(member['size_bytes']) is int and member['size_bytes']>=0
                          and digest(member['sha256']),'Invalid file tree member')
        _require(names==sorted(set(names)),'Tree members must have sorted unique paths')
        _require(_tree_hash(item['members'])==item['sha256'],'Inventory tree hash does not match its members')
    _require(paths==sorted(paths),'Inventory roots must have sorted paths')
    # Round-trip rejects non-JSON values without touching the selected files.
    return json.loads(json.dumps(dict(snapshot),allow_nan=False,ensure_ascii=False))


def verify_inputs(snapshot):
    """Rehash all recorded inputs; return a JSON success record or raise ValueError."""
    expected=_validate_snapshot(snapshot)
    actual=freeze_inputs(item['path'] for item in expected['entries'])
    if actual!=expected:
        for before,after in zip(expected['entries'],actual['entries']):
            if before!=after:raise ValueError('Frozen input changed: '+before['path'])
        raise ValueError('Frozen input inventory changed')
    files=sum(1 if item['kind']=='file' else sum(member['kind']=='file' for member in item['members'])
              for item in actual['entries'])
    return dict(schema_version=1,kind='input_sha256_verification',phase='verified',verified=True,
        hash_algorithm='sha256',tree_hash_format=_TREE_FORMAT,roots_verified=len(actual['entries']),
        files_verified=files,trees_verified=sum(item['kind']=='directory' for item in actual['entries']))
