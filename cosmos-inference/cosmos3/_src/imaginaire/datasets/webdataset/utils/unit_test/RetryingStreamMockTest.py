# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from http.client import IncompleteRead
from unittest.mock import MagicMock, patch

from botocore.exceptions import EndpointConnectionError, ResponseStreamingError
from urllib3.exceptions import ProtocolError as URLLib3ProtocolError
from urllib3.exceptions import ReadTimeoutError as URLLib3ReadTimeoutError
from urllib3.exceptions import SSLError as URLLib3SSLError

import cosmos3._src.imaginaire.datasets.webdataset.utils.stream as stream_module
from cosmos3._src.imaginaire.datasets.webdataset.utils.stream import RetryingStream

# Configure faster logging interval for tests (10 seconds instead of 5 minutes)
stream_module.RETRY_STATS_LOG_INTERVAL = 10.0

# Test 1: Simulate IncompleteRead and verify retry works
print("Test 1: Retry on IncompleteRead")
client = MagicMock()
expected_data = b"X" * 100  # 100 bytes of data
client.head_object.return_value = {"ContentLength": str(len(expected_data))}

# Create mock streams
mock_body_1 = MagicMock()
mock_body_1.close = MagicMock()  # Track if close() is called
mock_body_1.read.side_effect = IncompleteRead(b"partial")  # First read fails

mock_body_2 = MagicMock()
mock_body_2.read.return_value = expected_data  # Second read succeeds with full data

# Return different bodies on each get_object call
client.get_object.side_effect = [
    {"Body": mock_body_1, "ContentLength": len(expected_data)},  # First attempt
    {"Body": mock_body_2, "ContentLength": len(expected_data)},  # Retry attempt
]

stream = RetryingStream(client, "test-bucket", "test.tar", retries=3)

# Mock time.sleep to skip waiting
with patch("time.sleep"):
    data = stream.read(100)

assert data == expected_data, f"Expected {len(expected_data)} bytes but got {len(data)} bytes"
assert len(data) == 100, f"Expected 100 bytes but got {len(data)}"
assert mock_body_1.close.called, "Old stream was not closed"
assert client.get_object.call_count == 2, f"Expected 2 calls but got {client.get_object.call_count}"
print(f"✓ Read succeeded after retry: {len(data)} bytes")
print(f"✓ Old stream was closed: {mock_body_1.close.called}")
print(f"✓ get_object called {client.get_object.call_count} times (initial + retry)")

# Test 2: Multiple errors before success
print("\nTest 2: Multiple retries before success")
client2 = MagicMock()
expected_data2 = b"Y" * 200  # 200 bytes of data
client2.head_object.return_value = {"ContentLength": str(len(expected_data2))}

# Create multiple failing streams and one success
failing_bodies = []
for i in range(2):
    body = MagicMock()
    body.close = MagicMock()
    body.read.side_effect = IncompleteRead(b"fail")
    failing_bodies.append(body)

success_body = MagicMock()
success_body.read.return_value = expected_data2

client2.get_object.side_effect = [
    {"Body": failing_bodies[0], "ContentLength": len(expected_data2)},
    {"Body": failing_bodies[1], "ContentLength": len(expected_data2)},
    {"Body": success_body, "ContentLength": len(expected_data2)},
]

stream2 = RetryingStream(client2, "test-bucket", "test.tar", retries=5)

with patch("time.sleep"):
    data2 = stream2.read(200)

assert data2 == expected_data2, f"Expected {len(expected_data2)} bytes but got {len(data2)} bytes"
assert len(data2) == 200, f"Expected 200 bytes but got {len(data2)}"
assert failing_bodies[0].close.called, "First stream was not closed"
assert failing_bodies[1].close.called, "Second stream was not closed"
assert client2.get_object.call_count == 3, f"Expected 3 calls but got {client2.get_object.call_count}"
print(f"✓ Read succeeded after {client2.get_object.call_count - 1} retries: {len(data2)} bytes")
print(f"✓ First stream closed: {failing_bodies[0].close.called}")
print(f"✓ Second stream closed: {failing_bodies[1].close.called}")

# Test 3: Max retries exceeded
print("\nTest 3: Max retries exceeded")
client3 = MagicMock()
expected_size3 = 150
client3.head_object.return_value = {"ContentLength": str(expected_size3)}

# Always fail
always_fail_body = MagicMock()
always_fail_body.close = MagicMock()
always_fail_body.read.side_effect = IncompleteRead(b"always fail")

client3.get_object.return_value = {"Body": always_fail_body, "ContentLength": expected_size3}

stream3 = RetryingStream(client3, "test-bucket", "test.tar", retries=2)

exception_raised = False
try:
    with patch("time.sleep"):
        data3 = stream3.read(100)
    print("✗ Should have raised exception!")
    assert False, "Expected IncompleteRead exception to be raised"
except IncompleteRead:
    exception_raised = True
    print(f"✓ Correctly raised IncompleteRead after {stream3.retries} retries")

assert exception_raised, "Exception was not raised when it should have been"

# Test 4: Mix different error types
print("\nTest 4: Mixed error types (IncompleteRead + URLLib3ReadTimeoutError)")
client4 = MagicMock()
expected_data4 = b"Z" * 250  # 250 bytes
client4.head_object.return_value = {"ContentLength": str(len(expected_data4))}

error_body_1 = MagicMock()
error_body_1.close = MagicMock()
error_body_1.read.side_effect = IncompleteRead(b"incomplete")

error_body_2 = MagicMock()
error_body_2.close = MagicMock()
error_body_2.read.side_effect = URLLib3ReadTimeoutError(None, None, "timeout")

success_body_2 = MagicMock()
success_body_2.read.return_value = expected_data4

client4.get_object.side_effect = [
    {"Body": error_body_1, "ContentLength": len(expected_data4)},
    {"Body": error_body_2, "ContentLength": len(expected_data4)},
    {"Body": success_body_2, "ContentLength": len(expected_data4)},
]

stream4 = RetryingStream(client4, "test-bucket", "test.tar", retries=5)

with patch("time.sleep"):
    data4 = stream4.read(250)

assert data4 == expected_data4, f"Expected {len(expected_data4)} bytes but got {len(data4)} bytes"
assert len(data4) == 250, f"Expected 250 bytes but got {len(data4)}"
assert error_body_1.close.called, "First error stream was not closed"
assert error_body_2.close.called, "Second error stream was not closed"
assert client4.get_object.call_count == 3, f"Expected 3 calls but got {client4.get_object.call_count}"
print(f"✓ Recovered from mixed errors: {len(data4)} bytes")
print(f"✓ Both error streams were closed: {error_body_1.close.called and error_body_2.close.called}")

# Test 5: URLLib3ProtocolError
print("\nTest 5: Retry on URLLib3ProtocolError")
client5 = MagicMock()
expected_data5 = b"A" * 128
client5.head_object.return_value = {"ContentLength": str(len(expected_data5))}

error_body_5 = MagicMock()
error_body_5.close = MagicMock()
error_body_5.read.side_effect = URLLib3ProtocolError("Connection broken")

success_body_5 = MagicMock()
success_body_5.read.return_value = expected_data5

client5.get_object.side_effect = [
    {"Body": error_body_5, "ContentLength": len(expected_data5)},
    {"Body": success_body_5, "ContentLength": len(expected_data5)},
]

stream5 = RetryingStream(client5, "test-bucket", "test.tar", retries=3)

with patch("time.sleep"):
    data5 = stream5.read(128)

assert data5 == expected_data5, f"Expected {len(expected_data5)} bytes but got {len(data5)} bytes"
assert error_body_5.close.called, "Error stream was not closed"
assert client5.get_object.call_count == 2, f"Expected 2 calls but got {client5.get_object.call_count}"
print(f"✓ Recovered from ProtocolError: {len(data5)} bytes")

# Test 6: URLLib3SSLError
print("\nTest 6: Retry on URLLib3SSLError")
client6 = MagicMock()
expected_data6 = b"B" * 256
client6.head_object.return_value = {"ContentLength": str(len(expected_data6))}

error_body_6 = MagicMock()
error_body_6.close = MagicMock()
error_body_6.read.side_effect = URLLib3SSLError("SSL handshake failed")

success_body_6 = MagicMock()
success_body_6.read.return_value = expected_data6

client6.get_object.side_effect = [
    {"Body": error_body_6, "ContentLength": len(expected_data6)},
    {"Body": success_body_6, "ContentLength": len(expected_data6)},
]

stream6 = RetryingStream(client6, "test-bucket", "test.tar", retries=3)

with patch("time.sleep"):
    data6 = stream6.read(256)

assert data6 == expected_data6, f"Expected {len(expected_data6)} bytes but got {len(data6)} bytes"
assert error_body_6.close.called, "Error stream was not closed"
assert client6.get_object.call_count == 2, f"Expected 2 calls but got {client6.get_object.call_count}"
print(f"✓ Recovered from SSLError: {len(data6)} bytes")

# Test 7: Generic IOError
print("\nTest 7: Retry on generic IOError")
client7 = MagicMock()
expected_data7 = b"C" * 512
client7.head_object.return_value = {"ContentLength": str(len(expected_data7))}

error_body_7 = MagicMock()
error_body_7.close = MagicMock()
error_body_7.read.side_effect = IOError("Generic IO error")

success_body_7 = MagicMock()
success_body_7.read.return_value = expected_data7

client7.get_object.side_effect = [
    {"Body": error_body_7, "ContentLength": len(expected_data7)},
    {"Body": success_body_7, "ContentLength": len(expected_data7)},
]

stream7 = RetryingStream(client7, "test-bucket", "test.tar", retries=3)

with patch("time.sleep"):
    data7 = stream7.read(512)

assert data7 == expected_data7, f"Expected {len(expected_data7)} bytes but got {len(data7)} bytes"
assert error_body_7.close.called, "Error stream was not closed"
assert client7.get_object.call_count == 2, f"Expected 2 calls but got {client7.get_object.call_count}"
print(f"✓ Recovered from IOError: {len(data7)} bytes")

# Test 8: Premature end of stream detection
print("\nTest 8: Premature end of stream detection")
client8 = MagicMock()
expected_data8 = b"D" * 1024
client8.head_object.return_value = {"ContentLength": str(len(expected_data8))}

# First body returns empty when we expect data (premature end)
premature_body = MagicMock()
premature_body.close = MagicMock()
premature_body.read.return_value = b""  # Empty read when we expect data

success_body_8 = MagicMock()
success_body_8.read.return_value = expected_data8

client8.get_object.side_effect = [
    {"Body": premature_body, "ContentLength": len(expected_data8)},
    {"Body": success_body_8, "ContentLength": len(expected_data8)},
]

stream8 = RetryingStream(client8, "test-bucket", "test.tar", retries=3)

with patch("time.sleep"):
    data8 = stream8.read(1024)

assert data8 == expected_data8, f"Expected {len(expected_data8)} bytes but got {len(data8)} bytes"
assert premature_body.close.called, "Premature stream was not closed"
assert client8.get_object.call_count == 2, f"Expected 2 calls but got {client8.get_object.call_count}"
print(f"✓ Recovered from premature end of stream: {len(data8)} bytes")

# Test 9: EndpointConnectionError during reconnection
print("\nTest 9: EndpointConnectionError during reconnection (continues retry loop)")
client9 = MagicMock()
expected_data9 = b"E" * 2048
client9.head_object.return_value = {"ContentLength": str(len(expected_data9))}

# First read fails
error_body_9a = MagicMock()
error_body_9a.close = MagicMock()
error_body_9a.read.side_effect = URLLib3ReadTimeoutError(None, None, "timeout")

# First reconnection attempt fails with EndpointConnectionError
# Second read also fails
error_body_9b = MagicMock()
error_body_9b.close = MagicMock()
error_body_9b.read.side_effect = URLLib3ReadTimeoutError(None, None, "timeout")

# Final success
success_body_9 = MagicMock()
success_body_9.read.return_value = expected_data9

# Simulate EndpointConnectionError on first reconnection attempt, then succeed
client9.get_object.side_effect = [
    {"Body": error_body_9a, "ContentLength": len(expected_data9)},  # Initial read
    EndpointConnectionError(endpoint_url="https://s3.amazonaws.com"),  # Reconnection fails
    {"Body": error_body_9b, "ContentLength": len(expected_data9)},  # Second attempt after endpoint error
    {"Body": success_body_9, "ContentLength": len(expected_data9)},  # Final success
]

stream9 = RetryingStream(client9, "test-bucket", "test.tar", retries=5)

with patch("time.sleep"):
    data9 = stream9.read(2048)

assert data9 == expected_data9, f"Expected {len(expected_data9)} bytes but got {len(data9)} bytes"
assert error_body_9a.close.called, "First error stream was not closed"
assert error_body_9b.close.called, "Second error stream was not closed"
# Should be called 4 times: initial + endpoint error + retry after endpoint + final success
assert client9.get_object.call_count == 4, f"Expected 4 calls but got {client9.get_object.call_count}"
print(f"✓ Recovered from EndpointConnectionError during reconnection: {len(data9)} bytes")

# Test 10: ResponseStreamingError during reconnection (now FIXED)
print("\nTest 10: ResponseStreamingError during reconnection - should retry")
client10 = MagicMock()
expected_data10 = b"F" * 1024
client10.head_object.return_value = {"ContentLength": str(len(expected_data10))}

# First read fails with IncompleteRead
error_body_10a = MagicMock()
error_body_10a.close = MagicMock()
error_body_10a.read.side_effect = IncompleteRead(b"incomplete")

# Reconnection fails with ResponseStreamingError (wrapping IncompleteRead)
reconnect_error = ResponseStreamingError(
    error=IncompleteRead(b"x" * 97727), msg="Connection broken: IncompleteRead(97727 bytes read, 143937 more expected)"
)

# Second attempt after reconnection succeeds
success_body_10 = MagicMock()
success_body_10.read.return_value = expected_data10

# Simulate: read fails → reconnect fails with ResponseStreamingError → retry succeeds
client10.get_object.side_effect = [
    {"Body": error_body_10a, "ContentLength": len(expected_data10)},  # Initial read
    reconnect_error,  # First reconnection fails with ResponseStreamingError (now caught!)
    {"Body": success_body_10, "ContentLength": len(expected_data10)},  # Second reconnection succeeds
]

stream10 = RetryingStream(client10, "test-bucket", "test.tar", retries=5)

with patch("time.sleep"):
    with patch("cosmos3._src.imaginaire.datasets.webdataset.utils.stream.log"):
        data10 = stream10.read(1024)

# Verify the fix worked
assert data10 == expected_data10, f"Expected {len(expected_data10)} bytes but got {len(data10)} bytes"
assert error_body_10a.close.called, "First error stream was not closed"
assert client10.get_object.call_count == 3, f"Expected 3 calls but got {client10.get_object.call_count}"
print(f"✓ Recovered from ResponseStreamingError during reconnection: {len(data10)} bytes")

# Test 11: Failure during __init__ get_length() - now WITH retry logic
print("\nTest 11: Failure during __init__ get_length() - retries 3 times then fails")
client11 = MagicMock()

# head_object fails with ResponseStreamingError on all attempts
client11.head_object.side_effect = ResponseStreamingError(
    error=IncompleteRead(b"fail"), msg="Connection broken during head_object"
)

test11_exception = None
with patch("time.sleep"):  # Skip sleep delays in test
    try:
        stream11 = RetryingStream(client11, "test-bucket", "test.tar", retries=5)
        print("✗ Should have raised exception after retries exhausted")
    except ResponseStreamingError as e:
        test11_exception = e
        print(f"✓ ResponseStreamingError raised after retries exhausted")
        print(f"  Error: {e}")
        print(f"  head_object was called {client11.head_object.call_count} time(s)")

assert test11_exception is not None, "Should have raised exception after retries exhausted"
assert client11.head_object.call_count == 5, "Should have retried 5 times (retries=5)"


# Test 12: Failure during __init__ get_stream() - now WITH retry logic
print("\nTest 12: Failure during __init__ get_stream() - retries 3 times then fails")
client12 = MagicMock()
client12.head_object.return_value = {"ContentLength": "1024"}

# get_object fails with ResponseStreamingError during initial stream creation on all attempts
client12.get_object.side_effect = ResponseStreamingError(
    error=IncompleteRead(b"fail"), msg="Connection broken during initial get_object"
)

test12_exception = None
with patch("time.sleep"):  # Skip sleep delays in test
    try:
        stream12 = RetryingStream(client12, "test-bucket", "test.tar", retries=5)
        print("✗ Should have raised exception after retries exhausted")
    except ResponseStreamingError as e:
        test12_exception = e
        print(f"✓ ResponseStreamingError raised after retries exhausted")
        print(f"  Error: {e}")
        print(f"  get_object was called {client12.get_object.call_count} time(s)")

assert test12_exception is not None, "Should have raised exception after retries exhausted"
assert client12.get_object.call_count == 5, "Should have retried 5 times (retries=5)"


# Test 13: Transient failure during __init__ get_stream() on first attempt, success on retry
print("\nTest 13: Transient failure during __init__ - now succeeds with retry logic!")
client13 = MagicMock()
expected_data13 = b"G" * 512
client13.head_object.return_value = {"ContentLength": str(len(expected_data13))}

# First get_object fails, second succeeds (showing network blip during initialization)
success_body_13 = MagicMock()
success_body_13.read.return_value = expected_data13

get_object_call_count = [0]


def get_object_with_initial_failure(**kwargs):
    get_object_call_count[0] += 1
    if get_object_call_count[0] == 1:
        # First call during __init__ fails
        raise ResponseStreamingError(
            error=IncompleteRead(b"transient"), msg="Transient network error during initialization"
        )
    else:
        # Subsequent calls succeed
        return {"Body": success_body_13, "ContentLength": len(expected_data13)}


client13.get_object.side_effect = get_object_with_initial_failure

test13_exception = None
test13_stream = None
with patch("time.sleep"):  # Skip sleep delays in test
    try:
        test13_stream = RetryingStream(client13, "test-bucket", "test.tar", retries=5)
        print(f"✓ Object created successfully after transient failure")
        print(f"  get_object was called {get_object_call_count[0]} time(s)")
    except ResponseStreamingError as e:
        test13_exception = e
        print(f"✗ Unexpected failure: {e}")

# Verify the retry logic worked
assert test13_exception is None, "Should have succeeded after retry"
assert test13_stream is not None, "Stream should be created"
assert get_object_call_count[0] == 2, "Should have failed once, then succeeded on retry"
data13 = test13_stream.read()
assert data13 == expected_data13, "Should be able to read data successfully"
print(f"✓ Successfully read {len(data13)} bytes after recovering from transient init error")

print("\n✅ All mock tests passed! Retry logic working correctly.")
