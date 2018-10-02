import os
import sys
import multiprocessing
import textdistance as td


def build_paths_from_sys(args=None):
  dataset = sys.argv[1] if args is None else args[0]
  name = sys.argv[2] if args is None else str(args[1])

  if not name.endswith('.txt'):
    name = name + '.txt'

  return (
    os.path.join(
      dataset, name
    ),
    None
  )


def read_task_repair_solution(task_path, solution_path):
  with open(task_path, 'r') as tf:
    task = tf.readlines()

    assert len(task) > 2

    return (
      task[0].strip(),
      '',
      0,
      task[2:]
    )


def metrics(x):
  a = x[4].strip()
  b = x[5].strip()

  al = a.lower()
  bl = b.lower()

  a_len = float(len(a))

  def tryit(x):
    try:
      return x()
    except Exception as e:
      return 0.0

  tempo = lambda a, b, x: \
    sum([ 
      1 if xi == a else (-1 if xi == b else 0) for xi in x
    ])
  
  M = [
    x[3],
    tryit(lambda: td.bz2_ncd(a, b)),
    tryit(lambda: td.zlib_ncd(a, b)),
    tryit(lambda: td.prefix.normalized_similarity(a, b)),
    tryit(lambda: td.postfix.normalized_similarity(a, b)),
    tryit(lambda: td.matrix.normalized_similarity(a, b)),
    tryit(lambda: td.length.normalized_similarity(a, b)),
    tryit(lambda: td.Hamming().normalized_similarity(a, b)),
    tryit(lambda: td.Hamming(qval=2).normalized_similarity(a, b)),
    tryit(lambda: td.Hamming(qval=3).normalized_similarity(a, b)),
    tryit(lambda: td.Hamming(qval=4).normalized_similarity(a, b)),
    tryit(lambda: td.Hamming(qval=5).normalized_similarity(a, b)),
    tryit(lambda: td.DamerauLevenshtein().normalized_similarity(a, b)),
    tryit(lambda: td.DamerauLevenshtein(qval=2).normalized_similarity(a, b)),
    tryit(lambda: td.DamerauLevenshtein(qval=3).normalized_similarity(a, b)),
    tryit(lambda: td.DamerauLevenshtein(qval=4).normalized_similarity(a, b)),
    tryit(lambda: td.DamerauLevenshtein(qval=5).normalized_similarity(a, b)),
    tryit(lambda: td.Jaccard().normalized_similarity(a, b)),
    tryit(lambda: td.Jaccard().normalized_similarity(al, bl)),
    tryit(lambda: td.Jaccard(qval=2).normalized_similarity(a, b)),
    tryit(lambda: td.Jaccard(qval=2).normalized_similarity(al, bl)),
    tryit(lambda: td.Jaccard(qval=3).normalized_similarity(a, b)),
    tryit(lambda: td.Jaccard(qval=3).normalized_similarity(al, bl)),
    tryit(lambda: td.Jaccard(qval=4).normalized_similarity(a, b)),
    tryit(lambda: td.Jaccard(qval=4).normalized_similarity(al, bl)),
    tryit(lambda: td.Jaccard(qval=5).normalized_similarity(a, b)),
    tryit(lambda: td.Jaccard(qval=5).normalized_similarity(al, bl)),
    tryit(lambda: td.Tversky().normalized_similarity(a, b)),
    tryit(lambda: td.Tversky(qval=2).normalized_similarity(a, b)),
    tryit(lambda: td.Tversky(qval=3).normalized_similarity(a, b)),
    tryit(lambda: td.Tversky(qval=4).normalized_similarity(a, b)),
    tryit(lambda: td.Tversky(qval=5).normalized_similarity(a, b)),
    tryit(lambda: td.JaroWinkler().normalized_similarity(a, b)),
    tryit(lambda: td.JaroWinkler(qval=2).normalized_similarity(a, b)),
    tryit(lambda: td.JaroWinkler(qval=3).normalized_similarity(a, b)),
    tryit(lambda: td.JaroWinkler(qval=4).normalized_similarity(a, b)),
    tryit(lambda: td.JaroWinkler(qval=5).normalized_similarity(a, b)),
    tryit(lambda: td.StrCmp95().normalized_similarity(a, b)),
    tryit(lambda: td.StrCmp95().normalized_similarity(al, bl)),
    1.0 - (float(abs(tempo('(', ')', a) - tempo('(', ')', b))) / a_len),
    1.0 - (float(abs(tempo('[', ']', a) - tempo('[', ']', b))) / a_len),
    1.0 - (float(abs(tempo('{', '}', a) - tempo('{', '}', b))) / a_len),
    1.0 - (float(abs(tempo('<', '>', a) - tempo('<', '>', b))) / a_len)
  ]

  return '{} qid:{} {} # {}'.format(
    x[0],
    x[1],
    ' '.join([ 
      '{}:{:.4f}'.format(k+1, float(y)) for k,y in enumerate(M)
    ]),
    x[2]
  )


if __name__ == '__main__':
  print('[instauro]   Building dataset...', file=sys.stderr)
  
  CPUS = multiprocessing.cpu_count() // 2
  pool = multiprocessing.Pool(CPUS)
  targets = []

  tasks = os.listdir(sys.argv[1])

  for i in range(1, len(tasks) + 1):
    repair, solution, solution_line, source = read_task_repair_solution(
      *build_paths_from_sys([sys.argv[1], i])
    )

    repair = repair.strip()

    targets.extend([ 
      (
        0,
        i,
        j+1,
        float(j) / float(len(source)-2), 
        repair, 
        line.strip()
      ) for (j,line) in enumerate(source)
    ])

  print(
    '[instauro]   Processing dataset (using {} threads)...'.format(CPUS),
    file=sys.stderr
  )
  for r in pool.map(metrics, targets):
    print(r)

