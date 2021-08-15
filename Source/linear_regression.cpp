#include <vector>
#include <iostream>
#include <fstream>

using Radix = double;

template<class UnknownType>
class Linear_Regression {
public:
	constexpr void Clear() {
		(this->xData).clear();
		(this->yData).clear();
		(this->XX).clear();
		(this->Xy).clear();

		(this->Sigma_X) = 0.0;
		(this->Sigma_Y) = 0.0;
		(this->Sigma_XX) = 0.0;
		(this->Sigma_Xy) = 0.0;
		(this->W) = 0.0;
		(this->b) = 0.0;
		(this->N) = 0.0;
	}

	constexpr bool Train(std::vector<UnknownType>& xData, std::vector<UnknownType>& yData) {
		(this->xData) = xData;
		(this->yData) = yData;

		if (!calcN()) {
			return false;
		}

		Sigma();
		calcW();
		calcb();

		return true;
	}

	constexpr bool calcN() {
		if ((this->xData.size()) != (this->yData.size())
		|| ((this->xData.size()) == 0 && (this->yData.size()) == 0)) {
			return false;
		}

		(this->N) = (this->xData).size();

		return true;
	}

	constexpr void Sigma() {
		for (std::size_t i = 0; i < (this->N); ++i) {
			(this->XX).push_back((this->xData[i]) * (this->xData[i]));
			(this->Xy).push_back((this->xData[i]) * (this->yData[i]));
		}

		for (std::size_t i = 0; i < (this->N); ++i) {
			(this->Sigma_X) += (this->xData[i]);
			(this->Sigma_Y) += (this->yData[i]);
			(this->Sigma_XX) += (this->XX[i]);
			(this->Sigma_Xy) += (this->Xy[i]);
		}
	}

	constexpr void calcW() {
		(this->W) =
			((static_cast<UnknownType>(this->N) * (this->Sigma_Xy)) - ((this->Sigma_X) * (this->Sigma_Y))) /
			((static_cast<UnknownType>(this->N) * (this->Sigma_XX)) - ((this->Sigma_X) * (this->Sigma_X)));
	}

	constexpr void calcb() {
		(this->b) =
			((this->Sigma_Y) - ((this->W) * (this->Sigma_X))) /
			(static_cast<UnknownType>(this->N));
	}

	constexpr UnknownType Predict(std::size_t X) {
		return (((this->W) * X) + (this->b));
	}

	constexpr Radix getW() {
		return (this->W);
	}

	constexpr Radix getb() {
		return (this->b);
	}

	constexpr Linear_Regression() {
		Clear();
	}

private:
	std::vector<UnknownType> xData, yData;
	std::vector<UnknownType> XX, Xy;
	UnknownType Sigma_X, Sigma_Y, Sigma_XX, Sigma_Xy;

	std::size_t N;
	Radix W, b;
};

class Model_Logger {
public:
	bool Save(std::string Mode = "csv") {
		if (Mode == "csv") {
			FileName += ".csv";
		}
		else if (Mode == "txt") {
			FileName += ".txt";
		}
		else if (Mode == "lst") {
			FileName += ".lst";
		}
		else {
			return false;
		}

		std::ofstream ofs(FileName);

		if (!ofs.is_open()) {
			return false;
		}

		if (Mode == "csv") {
			ofs << "W,b\n";
			ofs << linear_regression->getW() << ',' << linear_regression->getb() << '\n';
		}
		else if (Mode == "txt") {
			ofs << linear_regression->getW() << ' ' << linear_regression->getb() << '\n';
		}
		else if (Mode == "lst") {
			ofs << linear_regression->getW() << ' ' << linear_regression->getb() << '\n';
		}

		return true;
	}
	
	bool Save(std::size_t const PredictingSize, std::string Mode = "csv") {
		if (Mode == "csv") {
			FileName += ".csv";
		}
		else if (Mode == "txt") {
			FileName += ".txt";
		}
		else if (Mode == "lst") {
			FileName += ".lst";
		}
		else {
			return false;
		}

		std::ofstream ofs(FileName);

		if (!ofs.is_open()) {
			return false;
		}

		if (Mode == "csv") {
			ofs << "W,b,Predict\n";
			ofs << linear_regression->getW() << ',' << linear_regression->getb() << ',' << linear_regression->Predict(PredictingSize) << '\n';
		}
		else if (Mode == "txt") {
			ofs << linear_regression->getW() << ' ' << linear_regression->getb() << ' ' << linear_regression->Predict(PredictingSize) << '\n';
		}
		else if (Mode == "lst") {
			ofs << linear_regression->getW() << ' ' << linear_regression->getb() << ' ' << linear_regression->Predict(PredictingSize) << '\n';
		}

		return true;
	}

	Model_Logger() {

	}

	template<class UnknownType>
	constexpr Model_Logger(Linear_Regression<UnknownType>* linear_regression, std::string FileName) {
		(this->linear_regression) = linear_regression;
		(this->FileName) = FileName;
	}

private:
	Linear_Regression<Radix>* linear_regression;
	std::string FileName;
};

class Initializer {
public:
	bool Init() {
		std::vector<Radix> xData =
		{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
		std::vector<Radix> yData =
		{ 1100, 2200, 3300, 4400, 5500, 6600, 7700, 8800, 9900, 11000 };

		if (!(this->linear_regression)->Train(xData, yData)) {
			return false;
		}
		
		constexpr std::size_t PredictingSize = 11;
		std::cout << ((this->linear_regression)->Predict(PredictingSize));

		model_logger->Save(PredictingSize, "txt");

		return true;
	}

	Initializer() {
		(this->linear_regression) = new Linear_Regression<Radix>;
		(this->model_logger) = new Model_Logger(linear_regression, "Model_Output");
	}

	~Initializer() {
		delete (this->linear_regression);
		delete (this->model_logger);
	}

private:
	Linear_Regression<Radix>* linear_regression;
	Model_Logger* model_logger;
};

int main() {
	std::ios_base::sync_with_stdio(0);
	std::cin.tie(0);
	std::cout.tie(0);

	Initializer* Init = new Initializer;

	if (!Init->Init()) {
		return -1;
	}

	delete Init;

	return 0;
}
