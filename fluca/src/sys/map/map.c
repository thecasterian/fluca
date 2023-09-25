#include <fluca/private/flucamapimpl.h>

#define FLUCA_MAP_INIT_BUCKET_SIZE 7
#define FLUCA_MAP_LOAD_FACTOR 0.75
#define FLUCA_MAP_GROWTH_FACTOR 1.5

PetscClassId FLUCA_MAP_CLASSID;

static PetscErrorCode FlucaMapCreateBuckets(FlucaMap map, PetscInt bucketsize) {
    PetscInt i;

    PetscFunctionBegin;

    PetscCheck(bucketsize > 0, PetscObjectComm((PetscObject)map), PETSC_ERR_ARG_OUTOFRANGE,
               "bucketsize must be positive");

    PetscCall(PetscMalloc1(bucketsize, &map->buckets));
    for (i = 0; i < bucketsize; i++) {
        map->buckets[i].front = NULL;
        map->buckets[i].back = NULL;
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FlucaMapKVListInsert(struct _FlucaMapKVList *list, struct _FlucaMapKV *kv) {
    PetscFunctionBegin;

    kv->prev = list->back;
    kv->next = NULL;
    if (list->back)
        list->back->next = kv;
    else
        list->front = kv;
    list->back = kv;

    PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FlucaMapKVListRemove(struct _FlucaMapKVList *list, struct _FlucaMapKV *kv) {
    PetscFunctionBegin;

    if (kv->prev)
        kv->prev->next = kv->next;
    else
        list->front = kv->next;
    if (kv->next)
        kv->next->prev = kv->prev;
    else
        list->back = kv->prev;

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaMapCreate(MPI_Comm comm, FlucaMap *map, PetscErrorCode (*hash)(PetscObject, PetscInt *),
                              PetscErrorCode (*eq)(PetscObject, PetscObject, PetscBool *)) {
    FlucaMap m;

    PetscFunctionBegin;

    *map = NULL;
    PetscCall(FlucaSysInitializePackage());

    PetscCall(FlucaHeaderCreate(m, FLUCA_MAP_CLASSID, "FlucaMap", "Map", "FlucaSys", comm, FlucaMapDestroy, NULL));

    PetscCall(FlucaMapCreateBuckets(m, FLUCA_MAP_INIT_BUCKET_SIZE));
    m->size = 0;
    m->bucketsize = FLUCA_MAP_INIT_BUCKET_SIZE;
    m->hash = hash;
    m->eq = eq;

    *map = m;

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaMapGetSize(FlucaMap map, PetscInt *size) {
    PetscFunctionBegin;
    PetscValidHeaderSpecific(map, FLUCA_MAP_CLASSID, 1);
    *size = map->size;
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaMapInsert(FlucaMap map, PetscObject key, PetscObject value) {
    PetscInt index;
    struct _FlucaMapKV *kv;

    PetscFunctionBegin;

    PetscValidHeaderSpecific(map, FLUCA_MAP_CLASSID, 1);
    PetscValidHeader(key, 2);
    PetscValidHeader(value, 3);

    PetscCall(PetscMalloc1(1, &kv));
    kv->key = key;
    kv->value = value;
    if (map->hash)
        PetscCall(map->hash(key, &kv->hash));
    else
        kv->hash = (uintptr_t)key % PETSC_INT_MAX;
    PetscCheck(kv->hash >= 0, PetscObjectComm((PetscObject)map), PETSC_ERR_ARG_OUTOFRANGE, "hash must be non-negative");

    PetscCall(PetscObjectReference((PetscObject)key));
    PetscCall(PetscObjectReference((PetscObject)value));

    index = kv->hash % map->bucketsize;
    PetscCall(FlucaMapKVListInsert(&map->buckets[index], kv));
    map->size++;

    if (map->size > FLUCA_MAP_LOAD_FACTOR * map->bucketsize) {
        struct _FlucaMapKVList *oldbuckets = map->buckets;
        PetscInt oldbucketsize = map->bucketsize;

        PetscCall(FlucaMapCreateBuckets(map, FLUCA_MAP_GROWTH_FACTOR * oldbucketsize));
        map->bucketsize = FLUCA_MAP_GROWTH_FACTOR * oldbucketsize;

        for (PetscInt i = 0; i < oldbucketsize; i++) {
            struct _FlucaMapKV *curr, *next;

            for (curr = oldbuckets[i].front; curr; curr = next) {
                next = curr->next;
                index = curr->hash % map->bucketsize;
                PetscCall(FlucaMapKVListInsert(&map->buckets[index], curr));
            }
        }
        PetscCall(PetscFree(oldbuckets));
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaMapRemove(FlucaMap map, PetscObject key) {
    PetscInt hash, index;
    struct _FlucaMapKV *kv;
    PetscBool eq;

    PetscFunctionBegin;

    PetscValidHeaderSpecific(map, FLUCA_MAP_CLASSID, 1);
    PetscValidHeader(key, 2);

    if (map->hash)
        PetscCall(map->hash(key, &hash));
    else
        hash = (uintptr_t)key % PETSC_INT_MAX;
    PetscCheck(hash >= 0, PetscObjectComm((PetscObject)map), PETSC_ERR_ARG_OUTOFRANGE, "hash must be non-negative");

    index = hash % map->bucketsize;
    for (kv = map->buckets[index].front; kv; kv = kv->next) {
        if (map->eq)
            PetscCall(map->eq(kv->key, key, &eq));
        else
            eq = kv->key == key;
        if (eq) {
            PetscCall(FlucaMapKVListRemove(&map->buckets[index], kv));
            PetscCall(PetscObjectDereference((PetscObject)kv->key));
            PetscCall(PetscObjectDereference((PetscObject)kv->value));
            PetscCall(PetscFree(kv));
            map->size--;
            break;
        }
    }

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaMapGetValue(FlucaMap map, PetscObject key, PetscObject *value) {
    PetscInt hash, index;
    struct _FlucaMapKV *kv;
    PetscBool eq;

    PetscFunctionBegin;

    PetscValidHeaderSpecific(map, FLUCA_MAP_CLASSID, 1);
    PetscValidHeader(key, 2);

    if (map->hash)
        PetscCall(map->hash(key, &hash));
    else
        hash = (uintptr_t)key % PETSC_INT_MAX;
    PetscCheck(hash >= 0, PetscObjectComm((PetscObject)map), PETSC_ERR_ARG_OUTOFRANGE, "hash must be non-negative");

    index = hash % map->bucketsize;
    for (kv = map->buckets[index].front; kv; kv = kv->next) {
        if (map->eq)
            PetscCall(map->eq(kv->key, key, &eq));
        else
            eq = kv->key == key;
        if (eq) {
            *value = kv->value;
            PetscFunctionReturn(PETSC_SUCCESS);
        }
    }

    *value = NULL;

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FlucaMapDestroy(FlucaMap *map) {
    PetscFunctionBegin;

    if (!*map)
        PetscFunctionReturn(PETSC_SUCCESS);
    PetscValidHeaderSpecific(*map, FLUCA_MAP_CLASSID, 1);

    if (--((PetscObject)(*map))->refct > 0) {
        *map = NULL;
        PetscFunctionReturn(PETSC_SUCCESS);
    }

    for (PetscInt i = 0; i < (*map)->bucketsize; i++) {
        struct _FlucaMapKV *curr, *next;

        for (curr = (*map)->buckets[i].front; curr; curr = next) {
            next = curr->next;
            PetscCall(PetscObjectDereference((PetscObject)curr->key));
            PetscCall(PetscObjectDereference((PetscObject)curr->value));
            PetscCall(PetscFree(curr));
        }
    }
    PetscCall(PetscFree((*map)->buckets));

    PetscCall(PetscHeaderDestroy(map));

    PetscFunctionReturn(PETSC_SUCCESS);
}
