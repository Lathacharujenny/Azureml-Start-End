{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "ad13090c",
   "metadata": {},
   "outputs": [],
   "source": [
    "import joblib\n",
    "from azureml.core.model import Model\n",
    "import numpy\n",
    "import json"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9d72492a",
   "metadata": {},
   "outputs": [],
   "source": [
    "def init():\n",
    "    model_path = Model.get_model_path('Logisticmodel.joblib')\n",
    "    model = joblib.load(model_path)\n",
    "\n",
    "def run(raw_data):\n",
    "    try:\n",
    "        data = np.array(json.loads(raw_data)['data'])\n",
    "        result = model.predict(data)\n",
    "        return json.dumps({'result': result.to_list()})\n",
    "    except Exception as e:\n",
    "        return json.dumps({'error': str(e)})"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.5"
  },
  "varInspector": {
   "cols": {
    "lenName": 16,
    "lenType": 16,
    "lenVar": 40
   },
   "kernels_config": {
    "python": {
     "delete_cmd_postfix": "",
     "delete_cmd_prefix": "del ",
     "library": "var_list.py",
     "varRefreshCmd": "print(var_dic_list())"
    },
    "r": {
     "delete_cmd_postfix": ") ",
     "delete_cmd_prefix": "rm(",
     "library": "var_list.r",
     "varRefreshCmd": "cat(var_dic_list()) "
    }
   },
   "types_to_exclude": [
    "module",
    "function",
    "builtin_function_or_method",
    "instance",
    "_Feature"
   ],
   "window_display": false
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
