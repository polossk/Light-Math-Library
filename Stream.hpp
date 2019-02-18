#pragma once

#include "base.hpp"

namespace lmlib
{
	struct Stream
	{
		inline void Wait() {}
		inline bool CheckIdle() { return true; }
	};
} // namespace lmlib